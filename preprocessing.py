# preprocessing.py

from muat.util import *
from  muat.reader import *
import os
import numpy as np
import pdb
import traceback
from tqdm import tqdm
import json
import glob
import subprocess
import pandas as pd
import gzip
from muat.download import download_reference

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


def get_segmentnt_center_embedding(sequence, tokenizer, model, device, layer=29):
    """
    Args:
        sequence (str): DNA sequence (already processed with get_context())
        tokenizer: HuggingFace tokenizer
        model: HuggingFace SegmentNT model
        max_length (int): padded token length (must satisfy (max_length - 1) % 4 == 0)
        layer (int): which layer to extract from (-1 = final)
    
    Returns:
        np.array: vector embedding at the center nucleotide
    """

    if type(sequence) is not str:
        pass
    else:
        sequence = [sequence]

    tokens = tokenizer(
        sequence,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )
    tokens = {k:v.to(device) for k,v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]           # (N, L, D)

    # lengths = tokens['attention_mask'].sum(1)       # (N,)
    # centers = ((lengths-1) // 2).long()                # (N,)
    # hidden = hidden[torch.arange(hidden.size(0)), centers-30:centers+31]  # (N, 61, D)
    # return hidden.mean(dim=1).cpu().numpy()  # (N, D)

    # Here we pick the “center” position in the *unpadded* seq.
    lengths = tokens['attention_mask'].sum(1)       # (N,)
    centers = ((lengths-1) // 2).long()                # (N,)

    # gather the per‐sample center embedding:
    batch_emb = hidden[torch.arange(hidden.size(0)), centers]  # (N, D)
    return batch_emb.cpu().numpy()

 

def extract_nucleotide_transformer_embedding(
    prev_buf, next_buf,
    ref_genome,
    tokenizer=None, model=None, device=None,
    context: int = 2406, # add  6 to make tokenized sequence to be odd => 300 [1] 300
    layer=21,
    
):
    """
    Use `get_context()` to generate mutation-aware sequence, then embed using transformer.

    Args:
        ref_genome (dict): Reference genome {chrom: sequence}
        tokenizer, model: HuggingFace nucleotide transformer
        context (int): Context window size (must be power of 2)
        prev_buf, next_buf: Current variant of interest + next neighboring variants (list of Variant)

    Returns:
        np.array: Central embedding vector
    """
    

    if not next_buf:
        raise ValueError("next_buf (list of neighboring variants) must be provided.")
    
    variant_record = next_buf[0]
    # Retrieve mutation-aware context
    mut_seq, ref_seq = get_context_seq(prev_buf, next_buf, ref_genome, context=context)

    if mut_seq is None:
        raise ValueError(f"Failed to extract context for {variant_record.chrom}:{variant_record.pos}")

    # Tokenize the sequence into k-mers and feed into model
    # kmer_size = 6
    # kmers = [seq[i:i + kmer_size] for i in range(len(seq) - kmer_size + 1)]
    # tokens = tokenizer(kmers, is_split_into_words=True, return_tensors="pt",
    #                    padding=True, truncation=True)
    mut_emb = get_segmentnt_center_embedding(
        sequence=mut_seq,
        tokenizer=tokenizer,
        model=model,
        device=device,
        # max_token_length=context, # make it 1 modulo (4)
        layer=layer  # or -1, LAYER to extract the embedding from GFM
    )

    ref_emb = get_segmentnt_center_embedding(
        sequence=ref_seq,
        tokenizer=tokenizer,
        model=model,
        device=device,
        # max_token_length=context, # make it 1 modulo (4)
        layer=layer  # or -1, LAYER to extract the embedding from GFM
    )
    
    return mut_emb[0], ref_emb[0]

    
def run_gfm_inference_contextual(
    chunk, gpu_id, ref_genome_path, tmp_dir,
    nucleotide_model_name, output_format="npz", num_mutation_threshold=25000, debug=False
):
    if debug:
        print(f"[DEBUG] Running GFM inference on GPU {gpu_id} with {len(chunk)} VCF files.")
    else:
        print(f"[INFO-{gpu_id}] Running GFM inference on GPU {gpu_id} with {len(chunk)} VCF files.")
    if len(chunk) == 0:
        print(f"[INFO-{gpu_id}] No VCF files for GPU {gpu_id}.")
        return
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Load tokenizer and model
    if not debug:
        # tokenizer = AutoTokenizer.from_pretrained(nucleotide_model_name, trust_remote_code=True)
        # device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        # model = AutoModel.from_pretrained(nucleotide_model_name, trust_remote_code=True).to(device=device)
        # model.eval()
        tokenizer = AutoTokenizer.from_pretrained(nucleotide_model_name, trust_remote_code=True)
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = AutoModelForMaskedLM.from_pretrained(nucleotide_model_name, trust_remote_code=True).to(device=device)
        model.eval()
    
    genome_ref = None
    print(f"[INFO-{gpu_id}] Loading reference genome")
    genome_ref = read_reference(ref_genome_path, verbose=False)
    
    
    context_window = 3000 # 5994 # 6 added for making mutation position positioned at the center
    layer = 5 # 21 ? 

    for path in chunk:
        variants = []
        with gzip.open(path, 'rt') as vcf_file:
            # type_snvs should be False to set variant vtype to "SNV", unless it assigns "ref > alt" to vtype
            vr = get_reader(vcf_file, type_snvs=False, pass_only=True)
            for variant in vr:
                if variant.chrom == VariantReader.EOF:
                    break
                variants.append(variant)

        variants.sort(key=lambda v: (v.chrom, v.pos))

        mut_embeddings, ref_embeddings, meta_data = [], [], []
        prev_buf, next_buf = deque(), deque()
        n = len(variants)
        
        if n > num_mutation_threshold:
            continue

        n_var = n_invalid = n_invalid_chrom = n_ok = 0
        warned_invalid_chrom = False
        next_buf_index = 0

        if debug:
            debug_file = f"debug_{os.path.basename(path)}.txt"
            df = open(debug_file, 'w')

        for i in tqdm(range(n), desc=f"GPU{gpu_id} sweepline"):
            v = variants[i]
            n_var += 1

            # !!!!!!!!!!!ERASE ME!!!!!!!!!!!!

            if genome_ref and  v.chrom not in genome_ref:
                if not warned_invalid_chrom:
                    sys.stderr.write(
                        f"Warning: a chromosome found in data not present in reference: {v.chrom}\n"
                        "Check your reference and VCF file compatibility.\n"
                    )
                    warned_invalid_chrom = True
                n_invalid_chrom += 1
                continue

            # Clean next_buf and repopulate it with variants within context_window
            while next_buf and (next_buf[0].chrom != v.chrom or next_buf[0].pos < v.pos):
                prev_buf.append(next_buf.popleft())

            # Clean prev_buf
            while prev_buf and (prev_buf[0].chrom != v.chrom or prev_buf[0].pos < v.pos - context_window):
                prev_buf.popleft()


            while next_buf_index < n and variants[next_buf_index].chrom == v.chrom and variants[next_buf_index].pos <= v.pos + context_window:
                next_buf.append(variants[next_buf_index])
                next_buf_index += 1

            assert v.chrom == next_buf[0].chrom and v.pos == next_buf[0].pos, \
                f"Unexpected variant order: {v.chrom}:{v.pos} vs {next_buf[0].chrom}:{next_buf[0].pos}"

            try:
                if debug:
                    if len(prev_buf) > 0 or len(next_buf) > 1:
                        print(f"\n> {v.chrom}:{v.pos} {v.ref}>{v.alt}")
                        for prev in prev_buf:
                            df.write(f"Prev: {prev.chrom}:{prev.pos} {prev.ref}->{prev.alt}\n")
                        for next in next_buf:
                            df.write(f"Next: {next.chrom}:{next.pos} {next.ref}->{next.alt}\n")
                    
                    mutated_seq, ref_seq = get_context_seq(
                        prev_buf, next_buf, genome_ref, context_window, debug=debug,
                    )
                    
                    # add "|" to the center of the mutated sequence
                    mutated_seq = mutated_seq[:context_window//2] + "|" + mutated_seq[context_window//2] + "|" + mutated_seq[context_window//2+1:]
                    ref_seq = ref_seq[:context_window//2] + "|" + ref_seq[context_window//2] + "|" + ref_seq[context_window//2+1:]
                    
                    mut_emb = np.ones(10)
                    ref_emb = np.ones(10)
                    
                    if len(prev_buf) > 0 or len(next_buf) > 1:
                        df.write(f"{v.chrom}\t{v.pos}\t{v.ref}\t{v.alt}\n{genome_ref[v.chrom][next_buf[0].pos - 1 - context_window//2:next_buf[0].pos - 1 + context_window//2]}\n{mutated_seq}\n{ref_seq}\n")

                    if i % 100 == 0:
                        df.flush()
                else:
                    with torch.no_grad():
                        
                        mut_emb, ref_emb = extract_nucleotide_transformer_embedding(
                            prev_buf, next_buf,
                            ref_genome=genome_ref,
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            context=context_window,
                            layer=layer
                        )
                
                n_ok += 1
                # if n_ok > 4000:
                #     break
            except Exception as e:
                print(f"[WARN] Failed embedding for {v.chrom}:{v.pos}\nError: {e}")
                # emb = torch.zeros( (context_window//6) + 1).numpy()
                n_invalid += 1
                continue

            mut_embeddings.append(mut_emb)
            ref_embeddings.append(ref_emb)
            meta_data.append((v.chrom, v.pos, v.ref, v.alt, v.sample_id))
        if len(mut_embeddings) == 0:
            print(f"[WARN] No valid embeddings for {path}. Skipping.")
            continue
        mut_embeddings = np.stack(mut_embeddings)
        ref_embeddings = np.stack(ref_embeddings)
        meta_data = np.array(meta_data, dtype=object)

        basename = os.path.basename(path).replace('.tsv.gz', "").replace(".vcf.gz", "") + f'.embedding.{output_format}'
        output_path = os.path.join(tmp_dir, basename)

        if debug:
            print(f"[DEBUG] Done")
            return 
            
        if output_format == "npz":
            np.savez_compressed(output_path, embeddings=mut_embeddings, ref_embeddings=ref_embeddings , meta=meta_data)
        elif output_format == "pt":
            torch.save({'embeddings': torch.tensor(mut_embeddings), 'ref_embeddings': torch.tensor(ref_embeddings), 'meta': meta_data}, output_path)
        else:
            raise ValueError("Unsupported output format")

        print(f"[INFO] Embeddings saved to {output_path}")



def combine_somagg_chunks_to_platekey(sample_folder,tmp_dir):
    '''
    sample_folder : list of all sample folders
    tmp_dir: directory after combining all chunks per sample
    '''

    tmp_dir = ensure_dirpath(tmp_dir)
    sample_folder = ensure_dirpath(sample_folder)
    sample_folder = multifiles_handler(sample_folder)

    for sampfold in sample_folder:
        all_chunk = glob.glob(sampfold + '*.tsv')
        pd_persample = pd.DataFrame()
        for perchunk in all_chunk:
            pd_read = pd.read_csv(perchunk,sep='\t',low_memory=False)
            pd_persample = pd.concat([pd_persample,pd_read])
        samp_id = pd_persample['sample'].iloc[0]
        pd_persample.to_csv(tmp_dir + get_sample_name(samp_id) + '.muat.tsv', sep='\t')

def filtering_somagg_vcf(all_somagg_chunks,tmp_dir):
    '''
    all_somagg_chunks : list of all somAgg chunk vcf files
    tmp_dir : directory after filtering somagg vcf
    '''
    header_line = ''

    fns = multifiles_handler(all_somagg_chunks)

    tmp_dir = ensure_dirpath(tmp_dir)
    
    for fn in fns:
        filename_only = get_sample_name(fn)
        exportdir = tmp_dir
        os.makedirs(exportdir,exist_ok=True)
        output_file = exportdir + filename_only + '.tsv'
        #pdb.set_trace()
        
        fvcf = open(output_file, "w")
        with gzip.open(fn, 'rb') as f:
            val_start = 0
            header_line = ''
            for i, l in enumerate(f):
                variable = l.decode('utf-8')

                if variable.startswith('##'):
                    header_line = header_line + variable
                if variable.startswith('#CHROM'):
                    val_start = 1
                    colm = variable.split('\t')
                    colm_front = colm[0:8]
                    fvcf.write('PLATEKEY\t')
                    fvcf.write('\t'.join(colm_front))
                    fvcf.write('\n')
                else:
                    if val_start == 1:
                        colm_value = variable.split('\t')
                        colm_front = colm[0:8]
                        colm_b = colm[9:]
                        colm_back = []
                        for sub in colm_b:
                            colm_back.append(sub.replace("\n", ""))

                        col_vcf = '\t'.join(colm_front + ['Platekey', 'Values'])
                        condition = ['0/0', './.']
                        colm_value_front = colm_value[0:8]
                        colm_value_back = colm_value[9:]

                        for i_c, i_value in enumerate(colm_value_back):
                            if i_value.startswith('0/0') or i_value.startswith('./.'):
                                pass
                            else:
                                if i_value.split(':')[9] == 'PASS':
                                    platekey = colm_back[i_c]
                                    fvcf.write(platekey)
                                    fvcf.write('\t')
                                    fvcf.write('\t'.join(colm_value_front))
                                    fvcf.write('\n')

        fvcf.close()

        '''
        pd_read = pd.read_csv(fn, sep="\t", comment='#',low_memory=False)
        pd_read = pd_read.iloc[:,0:8]
        pd_read.columns = ['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO']
        pd_read['PLATEKEY'] = filename_only

        pd_read = pd_read[['PLATEKEY','#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO']]
        pd_read.to_csv(output_file,sep='\t',index=False)
        '''

        #for somagg files
        #pd_read = pd.read_csv(output_file,sep='\t',low_memory=False)



def preprocessing_tsv38_tokenizing(tsv_file,genome_reference_38_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess tsv file with GRCh38 and tokenize the motif, pos, and ges
    '''
    tsv_file = multifiles_handler(tsv_file)
    preprocessing_tsv38(tsv_file,genome_reference_38_path,tmp_dir)

    all_preprocessed_vcf = []

    tmp_dir = ensure_dirpath(tmp_dir)

    for x in tsv_file:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #pdb.set_trace()
    tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf,tmp_dir)
    #pdb.set_trace()
    all_tokenized = []
    for x in all_preprocessed_vcf:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.muat.tsv'):
            all_tokenized.append(tmp_dir + get_sample_name(x) + '.muat.tsv')
    
    for x in all_tokenized:
        pd_file = pd.read_csv(x,sep='\t',low_memory=False)
        #pdb.set_trace()
        all_samples = pd_file['sample'].unique().tolist()

        for samp in all_samples:
            persamp = pd_file.loc[pd_file['sample']==samp]
            os.makedirs(tmp_dir + samp,exist_ok=True)
            persamp_path = ensure_dirpath(tmp_dir + samp)
            persamp.to_csv(persamp_path + get_sample_name(x) + '.tsv',sep='\t',index=False)
    #os.remove(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #os.remove(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

def preprocessing_vcf38_tokenizing(vcf_file,genome_reference_38_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess vcf file with GRCh38 and tokenize the motif, pos, and ges
    '''
    vcf_file = multifiles_handler(vcf_file)
    preprocessing_vcf38(vcf_file,genome_reference_38_path,tmp_dir)

    all_preprocessed_vcf = []

    tmp_dir = ensure_dirpath(tmp_dir)

    for x in vcf_file:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir  + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #pdb.set_trace()
    return tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf,tmp_dir)
    #pdb.set_trace()

def preprocessing_tsv38(tsv_file,genome_reference_38_path,tmp_dir,verbose=True):
    '''
    Preprocess tsv file with GRCh38
    '''
    genome_ref38 = read_reference(genome_reference_38_path, verbose=verbose)
    fns = multifiles_handler(tsv_file)
    fns = [resolve_path(x) for x in fns]

    for i, fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        # get motif
        f, sample_name = open_stream(fn)
        vr = SomAggTSVReader(f=f, pass_only=True, type_snvs=False)
        status('Writing mutation sequences...', verbose)
        process_input(vr, sample_name, None,tmp_dir,genome_ref38=genome_ref38,liftover=True,verbose=verbose)        
        f.close()

def preprocessing_vcf38(vcf_file,genome_reference_38_path,tmp_dir,verbose=True):
    '''
    Preprocess vcf file with GRCh38
    '''

    if not os.path.exists(genome_reference_38_path):
        print('reference file not found')
        genome_reference_dir = os.path.dirname(genome_reference_38_path)
        print('Downloading reference file to ' + genome_reference_dir)
        download_reference(genome_reference_dir,hg19=False,hg38=True)
        genome_reference_38_path = ensure_dirpath(genome_reference_dir) + 'hg38.fa.gz'

    genome_ref38 = read_reference(genome_reference_38_path, verbose=verbose)
    fns = multifiles_handler(vcf_file)
    fns = [resolve_path(x) for x in fns]

    for i, fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        get_motif_pos_ges(fn, None, tmp_dir, genome_ref38=genome_ref38, liftover=True, verbose=verbose)

def preprocessing_vcf_tokenizing(vcf_file,genome_reference_path,tmp_dir,dict_motif,dict_pos,dict_ges, verbose=True):
    '''
    Preprocess vcf file and tokenize the motif, pos, and ges
    '''
    vcf_file = multifiles_handler(vcf_file)
    vcf_file = [resolve_path(x) for x in vcf_file]

    # 파일 타입별로 분류
    vcf_files = [f for f in vcf_file if f.endswith(('.vcf', '.vcf.gz'))]
    maf_files = [f for f in vcf_file if f.endswith(('.maf', '.maf.gz'))]

    if vcf_files:
        preprocessing_vcf(vcf_files, genome_reference_path, tmp_dir, verbose=verbose)
    if maf_files:
        preprocessing_maf(maf_files, genome_reference_path, tmp_dir, verbose=verbose)
    
    # preprocessing_vcf(vcf_file,genome_reference_path,tmp_dir, verbose= verbose)
    # pdb.set_trace()
    all_preprocessed_vcf = []
    tmp_dir = ensure_dirpath(tmp_dir)
    
    for x in vcf_file:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
            
    #pdb.set_trace()
    return tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf,tmp_dir)
    

def get_motif_pos_ges(fn,genome_ref,tmp_dir,genome_ref38=None,liftover=False,verbose=True):
    """
    Preprocess to get the motif from the vcf file
    Args:
        fn: str, path to vcf file
        genome_ref: reference genome variable from read_reference
        tmp_dir: str, path to temporary directory for storing preprocessed files
        liftover: bool, if True, liftover the vcf file from GRCh38 to GRCh37
    """

    tmp_dir = ensure_dirpath(tmp_dir)

    try:
        # get motif
        f, sample_name = open_stream(fn)
        vr = get_reader(f)
        status('Writing mutation sequences...', verbose)
        process_input(vr, sample_name, genome_ref,tmp_dir,genome_ref38=genome_ref38,liftover=liftover,verbose=verbose)
        f.close()
        return 1
    except Exception as e:
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return 0
    

def preprocessing_vcf(vcf_file,genome_reference_path,tmp_dir,info_column=None,verbose=True):
    """
    Preprocess one or more VCF files
    Args:
        vcf_file: str or list of str, path(s) to VCF file(s)
        genome_reference_path: str, path to genome reference file
        tmp_dir: str, path to temporary directory for storing preprocessed files
    """

    # Check if reference files exist
    if not os.path.exists(genome_reference_path):
        print('[INFO] reference file not found')
        genome_reference_dir = os.path.dirname(genome_reference_path)
        print('[INFO] downloading reference file to ' + genome_reference_dir)
        download_reference(genome_reference_dir,hg19=True,hg38=False)
        genome_reference_path = ensure_dirpath(genome_reference_dir) + 'hg19.fa.gz'

    print('[INFO] Loading reference genome...')
    genome_ref = read_reference(genome_reference_path,verbose=verbose)   

    fns = multifiles_handler(vcf_file)
    fns = [resolve_path(x) for x in fns]
    #pdb.set_trace()
    tmp_dir = ensure_dirpath(tmp_dir)

    for i,fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        get_motif_pos_ges(fn,genome_ref,tmp_dir,verbose=verbose)

    return 1

def preprocessing_maf(maf_files, genome_reference_path, tmp_dir, verbose=True):
    """MAF 파일 전처리 (VCF 로직 재활용)"""
    if not os.path.exists(genome_reference_path):
        print('[INFO] reference file not found')
        # 참조 다운로드 로직...
        
    print('[INFO] Loading reference genome...')
    genome_ref = read_reference(genome_reference_path, verbose=verbose)   

    fns = multifiles_handler(maf_files)
    fns = [resolve_path(x) for x in fns]
    tmp_dir = ensure_dirpath(tmp_dir)

    for i, fn in enumerate(fns):
        get_motif_pos_ges_maf(fn, genome_ref, tmp_dir, verbose=verbose)

def get_motif_pos_ges_maf(fn, genome_ref, tmp_dir, verbose=True):
    """MAF 파일용 motif/pos/ges 추출"""
    tmp_dir = ensure_dirpath(tmp_dir)
    
    try:
        # MAF 리더 사용
        f, sample_name = open_stream(fn)
        vr = MAFReader(f=f, pass_only=True, type_snvs=False)
        status('Writing mutation sequences...', verbose)
        process_input(vr, sample_name, genome_ref, tmp_dir, verbose=verbose)        
        f.close()
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 0

def load_dict(dict_path):
    with open(dict_path, 'r') as f:
        dict_data = json.load(f)
    return dict_data['dict_motif'], dict_data['dict_pos'], dict_data['dict_ges']


def create_dictionary(prep_path,pos_bin_size=1000000,save_dict_path=None):
    '''
    Create a dictionary of preprocessed vcf file and histology abbreviation
    Args:
        prep_path: str or list of str, path(s) to preprocessed vcf file(s)
    '''   # Convert single file path to list for consistent handling
    if isinstance(prep_path, str):
        prep_path = [prep_path]

    dict_motif = set()
    dict_pos = set()    
    dict_ges = set()

    for path in tqdm(prep_path, desc="Generating token", unit="file"):
        #load tsv.gz file
        df = pd.read_csv(path, sep='\t',compression='gzip',low_memory=False)
        #get aliquot_id
        motif = set(df['seq'].to_list())
        ps = (df['pos'] / pos_bin_size).apply(np.floor).astype(int).astype(str)

        chrom = df['chrom'].astype(str)
        chrompos = chrom + '_' + ps
        df['chrompos'] = chrompos
        chrompos = df['chrompos'].unique()
        df['ges'] = df['genic'].astype(str) + '_' + df['exonic'].astype(str) + '_' + df['strand'].astype(str)
        ges = df['ges'].unique()

        df.to_csv(path, sep='\t',compression='gzip',index=False)

        dict_motif.update(motif)
        dict_pos.update(chrompos)
        dict_ges.update(ges)

    # Save all dictionaries as a single JSON file
    combined_dict = {
        'dict_motif': dict_motif,
        'dict_pos': dict_pos,
        'dict_ges': dict_ges
    }

    # Example of converting a set to a list before serialization
    if 'dict_motif' in combined_dict:
        combined_dict['dict_motif'] = list(combined_dict['dict_motif'])  # Convert set to list
    if 'dict_pos' in combined_dict:
        combined_dict['dict_pos'] = list(combined_dict['dict_pos'])  # Convert set to list
    if 'dict_ges' in combined_dict:
        combined_dict['dict_ges'] = list(combined_dict['dict_ges'])  # Convert set to list

    if save_dict_path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_dict_path), exist_ok=True)
        
        with open(save_dict_path, 'w') as f:
            json.dump(combined_dict, f)
    
    return dict_motif, dict_pos, dict_ges

def tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,tmp_dir,pos_bin_size=1000000, num_mutation_threshold=None):
    '''
    Tokenizing the motif, pos, and ges
    '''
    processed_file = []
    for path in tqdm(all_preprocessed_vcf, desc="Tokenizing", unit="file"):

        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t',compression='gzip',low_memory=False)
            if num_mutation_threshold and len(df) > num_mutation_threshold:
                continue  # Skip files with more mutations than the threshold
            ps = (df['pos'] / pos_bin_size).apply(np.floor).astype(int).astype(str)
            chrom = df['chrom'].astype(str)
            chrompos = chrom + '_' + ps
            df['chrompos'] = chrompos        
            df['ges'] = df['genic'].astype(str) + '_' + df['exonic'].astype(str) + '_' + df['strand'].astype(str)
    
            # if not exist remove from df 
            df = df.merge(dict_motif, left_on='seq', right_on='seq', how='left')
            df = df.merge(dict_pos, left_on='chrompos', right_on='chrompos', how='left')
            df = df.merge(dict_ges, left_on='ges', right_on='ges', how='left')


            df = df.dropna(subset=['triplettoken', 'postoken', 'gestoken'])

            token_file = ensure_dirpath(tmp_dir) + get_sample_name(path) + '.muat.tsv'
            df.to_csv(token_file, sep='\t',index=False)
            processed_file.append(token_file)
            # df.to_csv(token_file, sep='\t',compression='gzip',index=False)
    
    return processed_file


if __name__ == "__main__":
    # Example usage
    
    ref_genome_path = "/home/data_ssd/reference/GRCh37/human_g1k_v37.fasta"
    model_type = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    device = torch.device("cuda")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_type, trust_remote_code=True).to(device)
    print("Tokenizer and model loaded successfully.")
    print("Reading reference genome...")
    genome_ref = read_reference(ref_genome_path, verbose=False, target_chr="1")
    
    # Example variant
    chrom = "1"
    pos = 123456
    ref = genome_ref[chrom][pos - 1]  # 1-based index
    # Example variant record (you would typically get this from a VCF file)
    variant_record = Variant(chrom=chrom, pos=pos, ref=ref, alt="T")
    # Example buffers (you would typically get these from a VCF file)
    prev_buf = []
    next_buf = [variant_record]
    
    extract_nucleotide_transformer_embedding(
        prev_buf, next_buf,
        ref_genome=genome_ref,
        tokenizer=tokenizer,
        model=model,
        device=device,
        context=2000,
    )

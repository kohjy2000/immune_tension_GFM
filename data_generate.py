# data_generation.py (separate CPU and GFM stages with distinct functions)

import os
import glob
import gzip
import pandas as pd
from muat.preprocessing import preprocessing_vcf, preprocessing_vcf38, tokenizing, run_gfm_inference_contextual
from collections import defaultdict, deque
import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel
from muat.util import multifiles_handler, ensure_dirpath, get_sample_name, read_codes
from muat.reader import read_reference, Variant, get_reader, VariantReader, get_context_seq
from multiprocessing import Pool
from functools import partial
import math
import pdb
import sys

def chrom_key(chrom: str):
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    try:
        return int(chrom)
    except ValueError:
        return {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(chrom, 100)

def concat_vcfs_merge_headers_sorted(input_paths, output_path):
    header_lines, column_header = [], None
    for p in input_paths:
        if p.endswith('.gz'):
            open_func = gzip.open
        else:
            open_func = open

        with open_func(p, 'rt') as f:
            for line in f:
                if line.startswith('##'):
                    header_lines.append(line)
                elif line.startswith('#CHROM'):
                    column_header = line
                    break
    if column_header is None:
        raise RuntimeError("No #CHROM header found in inputs.")
    header_lines = sorted(set(header_lines))
    records = []
    for p in input_paths:
        if p.endswith('.gz'):
            open_func = gzip.open
        else:
            open_func = open
        with open_func(p, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                cols = line.split('\t')
                records.append((cols[0], int(cols[1]), line))
    records.sort(key=lambda x: (chrom_key(x[0]), x[1]))
    if output_path.endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open
    with open_func(output_path, 'wt') as out:
        for hl in header_lines: out.write(hl)
        out.write(column_header)
        for _, _, rec in records: out.write(rec)
    
def generate_mutated_sequence(variant, prev_buf, next_buf, genome_ref, context):
    """Generate mutated reference sequence around variant."""
    chrom = variant.chrom
    center = variant.pos
    start = center - (context // 2)
    end = center + (context // 2)

    ref_seq = genome_ref[chrom][start-1:end].upper()  # -1 because 1-based to 0-based
    seq = list(ref_seq)

    # Apply all mutations in the context window
    for var in list(prev_buf) + list(next_buf):
        if var.chrom != chrom:
            continue
        rel_pos = var.pos - start
        if 0 <= rel_pos < len(seq):
            if len(var.ref) == 1 and len(var.alt) == 1:
                # SNP
                seq[rel_pos] = var.alt.upper()
            else:
                # You can extend to handle indels if needed
                seq[rel_pos] = f"[{var.ref}>{var.alt}]"

    seq_str = ''.join(seq)
    # Mark center variant for clarity
    center_idx = context // 2
    marked_seq = seq_str[:center_idx] + "|" + seq_str[center_idx] + "|" + seq_str[center_idx+1:]
    return marked_seq

def _merge_chunk_worker(chunk_items, tmp_dir):
    outps = []
    for sid, paths in chunk_items:
        outp = os.path.join(tmp_dir, f"{sid}.merged.vcf.gz") if len(paths) > 1 else paths[0]
        if len(paths) > 1:
            concat_vcfs_merge_headers_sorted(paths, outp)
        outps.append(outp)
    return outps



def cpu_preprocess_only(
    vcf_files, ref_genome_path, tmp_dir,
    dict_motif, dict_pos, dict_ges, num_workers,
    num_mutation_threshold, hg38=False
):
    vcf_files = multifiles_handler(vcf_files)
    tmp_dir = ensure_dirpath(tmp_dir)
    
    # Step 1: Split VCF files into num_workers batches for preprocessing
    # (since preprocessing_vcf has startup costs and can handle batches)
    files_per_worker = math.ceil(len(vcf_files) / num_workers)
    vcf_batches = [
        vcf_files[i:i + files_per_worker] 
        for i in range(0, len(vcf_files), files_per_worker)
    ]
    
    # Remove empty batches
    vcf_batches = [batch for batch in vcf_batches if batch]
    
    # Create a partial function for batch preprocessing and tokenizing
    process_and_tokenize_batch_func = partial(
        preprocessing_vcf_tokenizing_batch,
        ref_genome_path=ref_genome_path,
        tmp_dir=tmp_dir,
        dict_motif=dict_motif,
        dict_pos=dict_pos,
        dict_ges=dict_ges,
        hg38=hg38
    )
    
    # Process and tokenize VCF batches in parallel
    with Pool(processes=len(vcf_batches)) as pool:  # Use actual number of batches
        batch_results = pool.map(process_and_tokenize_batch_func, vcf_batches)
    
    # Flatten the results from all batches
    all_tokenized_results = []
    for batch_result in batch_results:
        if batch_result:
            all_tokenized_results.extend(batch_result)
    
    return all_tokenized_results


def preprocessing_vcf_tokenizing_batch(vcf_batch, ref_genome_path, tmp_dir, dict_motif, dict_pos, dict_ges, hg38=False):
    """
    Process a batch of VCF files using your existing preprocessing_vcf function,
    then tokenize each resulting file individually
    """
    try:
        # Resolve paths for the batch
        vcf_batch = [x for x in vcf_batch]
        
        # Use your existing preprocessing_vcf function that handles batches
        if hg38:
            preprocessing_vcf38(vcf_batch, ref_genome_path, tmp_dir, verbose=False)
        else:
            preprocessing_vcf(vcf_batch, ref_genome_path, tmp_dir, verbose=False)
        
        
        # Collect all successfully preprocessed files from this batch
        all_preprocessed_vcf = []
        tmp_dir = ensure_dirpath(tmp_dir)
        
        for vcf_file in vcf_batch:
            expected_file = tmp_dir + get_sample_name(vcf_file) + '.gc.genic.exonic.cs.tsv.gz'
            if os.path.exists(expected_file):
                all_preprocessed_vcf.append(expected_file)
        
        if not all_preprocessed_vcf:
            return []
        
        # Tokenize each preprocessed file individually
        # (since tokenizing doesn't have startup costs)
        tokenized_results = []
        for preprocessed_file in all_preprocessed_vcf:
            try:
                result = tokenizing(dict_motif, dict_pos, dict_ges, [preprocessed_file], tmp_dir)
                if result is not None:
                    tokenized_results.append(result)
            except Exception as e:
                print(f"Error tokenizing {preprocessed_file}: {e}")
        
        return tokenized_results
        
    except Exception as e:
        print(f"Error processing VCF batch {vcf_batch}: {e}")
        return []

   




def gfm_embedding_inference(
    merged_vcfs, ref_genome_path, tmp_dir,
    dict_motif, dict_pos, dict_ges,
    num_gpus, hg38, nucleotide_model_name, num_mutation_threshold, debug=False
):
    chunks = [merged_vcfs[i::num_gpus] for i in range(num_gpus)]
    
    assert len(chunks) == num_gpus, f"Expected {num_gpus} chunks, got {len(chunks)}"

    with mp.get_context('spawn').Pool(num_gpus) as pool:
        pool.starmap(
            run_gfm_inference_contextual,
            [(chunk, gpu_id % 4, ref_genome_path, tmp_dir,
              nucleotide_model_name, 
              "npz", num_mutation_threshold, debug)
             for gpu_id, chunk in enumerate(chunks)]
        )

def create_training_dataset(
    input_dirs,
    tmp_dir,
    ref_genome_path,
    dict_motif_path,
    dict_pos_path,
    dict_ges_path,
    patterns= ["*.vcf.gz"],
    num_workers: int = 4,
    hg38: bool = False,
    nucleotide_model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", #"InstaDeepAI/segment_nt", #"InstaDeepAI/nucleotide-transformer-v2.5-500m"
    num_mutation_threshold: int = 25000,
    debug: bool = False
):
    os.makedirs(tmp_dir, exist_ok=True)
    dict_motif = pd.read_csv(dict_motif_path, sep='\t')
    dict_pos   = pd.read_csv(dict_pos_path, sep='\t')
    dict_ges   = pd.read_csv(dict_ges_path, sep='\t')

    # vcf_files = []
    # for d in input_dirs:
    #     for p in patterns:
    #         vcf_files.extend(glob.glob(os.path.join(d, p)))
        
    # assert len(vcf_files) > 0, "No VCF files found. Check input directories and patterns."
    # print(f"[+] Found {len(vcf_files)} VCF files.")

    # grouped = defaultdict(list)
    # for v in vcf_files:
    #     sid = os.path.basename(v).split('.')[0]
    #     grouped[sid].append(v)

    # items = list(grouped.items())
    # chunks = [items[i::num_workers] for i in range(num_workers)]
    # merged_vcfs = []
    # with mp.get_context('spawn').Pool(num_workers) as exe:
    #     futures = [exe.apply_async(_merge_chunk_worker, args=(chunk, tmp_dir)) for chunk in chunks if chunk]
    #     for fut in tqdm.tqdm(futures, total=len(futures), desc="Merging VCF chunks"):
    #         merged_vcfs.extend(fut.get())
    # print(f"[+] {len(merged_vcfs)} VCFs after merging.")

    # read merged_vcfs from the folder by searching "*.merged.vcf.gz"
    merged_vcfs = sorted(glob.glob(os.path.join(tmp_dir, "*.merged.vcf.gz")))
    if not merged_vcfs:
        raise RuntimeError("No merged VCF files found in the output directory. Check the merging step.")

    # if not debug:
    #     print("[Stage 1] Preprocessing to tokens with CPU only...")
    #     cpu_preprocess_only(
    #         merged_vcfs,
    #         ref_genome_path,
    #         tmp_dir,
    #         dict_motif,
    #         dict_pos,
    #         dict_ges,
    #         num_workers,
    #         num_mutation_threshold,
    #         hg38=hg38 
    #     )

    print("[Stage 2] GFM inference with GPU...")
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Found {num_gpus} GPUs.")
    gfm_embedding_inference(
        merged_vcfs,
        ref_genome_path,
        tmp_dir,
        dict_motif,
        dict_pos,
        dict_ges,
        8 if not debug else 1, #num_gpus * 2
        hg38,
        nucleotide_model_name,
        num_mutation_threshold,
        debug=debug
    )

    print("[+] Two-stage preprocessing complete.")



if __name__ == "__main__":
    num_mutation_threshold = 20000
    DEBUG           = False  # Set to True for debugging mode
    dataset         = 'inocras'
    if dataset == "pcawg":
        hg38 = False
        input_dirs =[
            "/home/data_ssd/data_PCAWG/PCAWG-1/final_consensus_12oct_passonly/snv_mnv/",
            "/home/data_ssd/data_PCAWG/PCAWG-1/final_consensus_12oct_passonly/indel/"
        ]
        patterns= ["*.vcf.gz"]
        tmp_dir         = "/home/data_ssd/cancerfoundationmodel/muat/data/inputs_preprocessed_pcawg_Norm_RefMut/"
    elif dataset == "inocras":
        hg38 = True
        input_dirs = [
            "/home/data_ssd/data_breast_cohort/03_somatic/"
        ]
        patterns= ["*INDs_quick_fi.vcf", "*SNVs_quick_fi.vcf"]
        tmp_dir         = "/home/data_ssd/cancerfoundationmodel/muat/data/inputs_preprocessed_Inocras/"
    elif dataset == "test":
        hg38 = False
        input_dirs = [
            "/home/data_ssd/cancerfoundationmodel/muat/data/test_vcfs/"
        ]
        patterns= ["*INDs_quick_fi.vcf", "*SNVs_quick_fi.vcf"]
        tmp_dir         = "/home/data_ssd/cancerfoundationmodel/muat/data/test_preprocessed/"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if hg38:
        ref_genome_path = "/home/data_ssd/reference/GRCh38/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fasta"
    else:
        ref_genome_path = "/home/data_ssd/reference/GRCh37/human_g1k_v37.fasta"
    dict_motif_path = "/home/data_ssd/cancerfoundationmodel/muat/muat/extfile/dictMutation.tsv"
    dict_pos_path   = "/home/data_ssd/cancerfoundationmodel/muat/muat/extfile/dictChpos.tsv"
    dict_ges_path   = "/home/data_ssd/cancerfoundationmodel/muat/muat/extfile/dictGES.tsv"
    num_workers      = 30
    # --------------
    
    os.makedirs(tmp_dir, exist_ok=True)

    # validate dirs
    for p in input_dirs + [tmp_dir,
                           os.path.dirname(ref_genome_path),
                           os.path.dirname(dict_motif_path),
                           os.path.dirname(dict_pos_path),
                           os.path.dirname(dict_ges_path)
                           ]:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing directory: {p}")

    create_training_dataset(
        input_dirs,
        tmp_dir,
        ref_genome_path,
        dict_motif_path,
        dict_pos_path,
        dict_ges_path,
        patterns=patterns,
        hg38=hg38,  # Set to True if using hg38 reference
        num_workers=num_workers,
        num_mutation_threshold=num_mutation_threshold,  # Adjust as needed
        debug=DEBUG
    )
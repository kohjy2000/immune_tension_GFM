import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
import os
import pandas as pd
import pdb
import numpy as np
import math
import pickle
import random
from sklearn.utils import shuffle

class DataloaderConfig:

    def __init__(self, **kwargs):
        # Set default value for sampling_replacement
        self.sampling_replacement = False
        # Update with any provided values
        for k,v in kwargs.items():
            setattr(self, k, v)

class MuAtDataloader(Dataset):
    def __init__(self, data_split_tsv, config, same_sampling=False, get_embedding=False, liftover=False):
        self.config = config
        self.data_split_tsv = data_split_tsv
        self.model_input = config.model_input
        self.mutation_type_ratio = config.mutation_type_ratio
        self.mutation_sampling_size = config.mutation_sampling_size
        self.same_sampling = same_sampling
        self.sampling_replacement = config.sampling_replacement
        self.get_embedding = get_embedding
        self.liftover = liftover
        if self.liftover:
            from pyliftover import LiftOver
            from pkg_resources import resource_filename

            liftover_chain = resource_filename('muat', 'pkg_data/genomic_tracks/hg38ToHg19.over.chain.gz')
            self.lo = LiftOver(liftover_chain)
            
    def __len__(self):
        return len(self.data_split_tsv)
    
    def __getitem__(self, idx):
        return self.get_data(idx)

    def count_ratio(self,pd_row):
        row_count_init = {'SNV':0,'MNV':0,'indel':0,'SV/MEI':0,'Neg':0}
        count = pd_row.groupby('mut_type').size().to_dict()
        for key,value in count.items():
            row_count_init[key] = value
            if key == 'SV':
                row_count_init['SV/MEI'] += value
            elif key == 'MEI':
                row_count_init['SV/MEI'] += value

        mut_ratio = np.array(list(self.mutation_type_ratio.values()))
        avail_count = mut_ratio * self.mutation_sampling_size   
        row_count = np.array(list(row_count_init.values()))
            
        diff = avail_count - row_count
        pos = diff>0
        avail_count1 = row_count * pos
        diff = row_count > avail_count

        avail_count2 = avail_count * diff
        avail_count3 = avail_count1 + avail_count2
        shadowavail_count3 = avail_count3
        shadowavail_count3[0] = row_count[0]

        if sum(shadowavail_count3) > self.mutation_sampling_size:
            diff = self.mutation_sampling_size - sum(avail_count3) 
            shadowavail_count3[0] = diff + avail_count3[0]
            
        avail_count2 = shadowavail_count3.astype(int)

        if avail_count2[0]<0:

            secondmax = avail_count2[np.argmax(avail_count2)]
            avail_count2 = avail_count2 * 0.7

            avail_count = avail_count2

            diff = avail_count - row_count
            pos = diff>0
            avail_count1 = row_count * pos
            diff = row_count > avail_count

            avail_count2 = avail_count * diff
            avail_count3 = avail_count1 + avail_count2
            shadowavail_count3 = avail_count3
            shadowavail_count3[0] = row_count[0]

            if sum(shadowavail_count3) > self.mutation_sampling_size:
                diff = self.mutation_sampling_size - sum(avail_count3) 
                shadowavail_count3[0] = diff + avail_count3[0]
                
            avail_count2 = shadowavail_count3.astype(int)

        avail_count = avail_count2

        avail_count_dict = {}

        for i,key in enumerate(row_count_init.keys()):
            avail_count_dict[key] = avail_count[i]

        return avail_count_dict

    def get_data(self, idx):
        instances = self.data_split_tsv.iloc[idx]
        # pd_row = pd.read_csv(instances['prep_path'], sep='\t', compression='gzip', low_memory=False)
        pd_row = pd.read_csv(instances['prep_path'], sep='\t', low_memory=False)
        sample_path = instances['prep_path']
        if self.get_embedding:
            # embedding saved with np.savez_compressed(output_path, embeddings=embeddings, meta=meta_data)
            """
                    emb = extract_nucleotide_transformer_embedding(
                                prev_buf, next_buf,
                                ref_genome=genome_ref,
                                tokenizer=tokenizer,
                                model=model,
                                device=device,
                                context=context_window
                            )
                        # emb = np.ones(10)
                        
                        # if prev_buf:
                        #     print(f"Prev {len(prev_buf)}: {prev_buf[0].chrom}:{prev_buf[0].pos} {prev_buf[0].ref} -> {prev_buf[0].alt}")
                        # if next_buf:
                        #     print(f"Next {len(next_buf)}: {next_buf[0].chrom}:{next_buf[0].pos} {next_buf[0].ref} -> {next_buf[0].alt}")
                        
                        n_ok += 1
                    except Exception as e:
                        print(f"[WARN] Failed embedding for {v.chrom}:{v.pos} — {e}")
                        emb = torch.zeros( (context_window//6) + 1).numpy()
                        n_invalid += 1
                        continue

                    embeddings.append(emb)
                    meta_data.append((v.chrom, v.pos, v.ref, v.alt, v.sample_id))

                embeddings = np.stack(embeddings)
                meta_data = np.array(meta_data, dtype=object)

                basename = os.path.basename(path).replace('.tsv.gz', "").replace(".vcf.gz", "") + f'.embedding.{output_format}'
                output_path = os.path.join(tmp_dir, basename)
            """
            embedding_path = instances['embedding_path']
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(f"Embedding file {embedding_path} does not exist.")
            with np.load(embedding_path, allow_pickle=True) as data:
                embeddings = data['embeddings']
                ref_embeddings = data['ref_embeddings']
                meta_data = data['meta']
                meta_data = pd.DataFrame(meta_data, columns=['chrom', 'pos', 'ref', 'alt', 'sample_id'])
            # make pd dataframe that has embedding and metadata as row
            
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
                ref_embeddings = torch.tensor(ref_embeddings, dtype=torch.float32)
            else:
                raise ValueError(f"Embeddings in {embedding_path} are not a valid numpy array.")
            
            if self.liftover:
                def convert_coordinate_safe(row):
                    try:
                        # Make sure chromosome is properly formatted
                        chrom = str(row['chrom'])
                        if not chrom.startswith('chr'):
                            chrom = f'chr{chrom}'
                        
                        # Convert position to integer
                        pos = int(row['pos'])
                        
                        # Perform liftover
                        result = self.lo.convert_coordinate(chrom, pos)
                        # Return the result (it's already a list or None)
                        return result[0][0][3:], int(result[0][1])
                    except Exception as e:
                        # print(f"Error converting {row.get('chrom', 'unknown')}:{row.get('pos', 'unknown')}: {e}")
                        return None, None

                # meta_data = pd.DataFrame(meta_data, columns=['chrom', 'pos', 'ref', 'alt', 'sample_id'])
                meta_data[['chrom', 'pos']] = meta_data.apply(lambda row: convert_coordinate_safe(row), axis=1, result_type='expand')
                meta_data = meta_data[meta_data['pos'].notna()]
                meta_data['pos'] = meta_data['pos'].astype(int)  # Ensure pos is integer

            # build dictionary that has chr, pos, ref, alt as keys and values from meta_data and has value the embeddings
            data_embeddings = {
            }
            data_ref_embeddings = {

            }
            for i, (chrom, pos, ref, alt, sample_id) in enumerate(meta_data.values):
                data_embeddings[(chrom, int(pos), ref, alt)] = embeddings[i]
                data_ref_embeddings[(chrom, int(pos), ref, alt)] = ref_embeddings[i]


        # Get idx_class and idx_subclass if they exist
        idx_class = None
        if 'class_index' in instances.index.to_list():
            idx_class = torch.tensor(np.array(instances['class_index']), dtype=torch.long)
        
        idx_subclass = None
        if 'subclass_index' in instances.index.to_list():
            idx_subclass = torch.tensor(np.array(instances['subclass_index']), dtype=torch.long)
        
        # Sampling logic
        
        pd_sampling = pd.DataFrame()
        grab_col = []

        if self.model_input['motif']:
            grab_col.append('triplettoken')
        if self.model_input['pos']:
            grab_col.append('postoken')
        if self.model_input['ges']:
            grab_col.append('gestoken')

        

        if self.get_embedding:
            before_filter = len(pd_row)
            # print(f"Before drop na: {len(pd_row)} rows")
            pd_row = pd.merge(pd_row, meta_data, on=['chrom', 'pos', 'ref', 'alt'], how='left')
            
            pd_row['embedding'] = pd_row.apply(
                lambda row: data_embeddings.get((row['chrom'], row['pos'], row['ref'], row['alt']), None), axis=1
            )
            pd_row['ref_embedding'] = pd_row.apply(
                lambda row: data_ref_embeddings.get((row['chrom'], row['pos'], row['ref'], row['alt']), None), axis=1
            )
            grab_col.append('embedding')
            grab_col.append('ref_embedding')
            
            pd_row = pd_row[pd_row['embedding'].notna()]
            # print(f"After drop na: {len(pd_row)} rows")
            after_filter = len(pd_row)
            if (before_filter - after_filter)/after_filter > 0.3:
                print(f"Warning: More than 30% of embeddings are missing in {embedding_path}. Before: {before_filter}, After: {after_filter}")
            assert len(pd_row) > 0, f"No valid embeddings in sample {embedding_path}"
        try:
            avail_count = self.count_ratio(pd_row)
        except Exception as e:
            raise ValueError(f"Error {e}\n idx:{idx} sample_path: {sample_path} ")

        for key, value in avail_count.items():
            if value > 0:
                # print(key,value)
                pd_samp = pd_row[pd_row['mut_type'] == key].sample(n=value, replace=False)
                # add embedding to pd_samp, where embedding can be found in data_embeddings
                # print("pd_samp")
                # print(pd_samp)
                pd_sampling = pd.concat([pd_sampling, pd_samp[grab_col]], ignore_index=True)
        
        

        # Handle padding
        if self.sampling_replacement:
            np_triplettoken = pd_sampling.to_numpy()
            mins = self.mutation_sampling_size - len(np_triplettoken)
            pd_rest_sampling = pd_sampling.sample(n=mins, replace=True)
            pd_sampling = pd.concat([pd_sampling, pd_rest_sampling], ignore_index=True)
            datanumeric = torch.tensor(pd_sampling.to_numpy().T, dtype=torch.long)
        else:
            np_triplettoken = pd_sampling.to_numpy()
            is_padding = len(pd_sampling) < self.mutation_sampling_size
            mins = self.mutation_sampling_size - len(np_triplettoken) if is_padding else 0

            datanumeric = []
            # Handle embedding first
            # if self.get_embedding:
            #     emb_list = pd_sampling['embedding'].tolist()
            #     if is_padding:
            #         pad_tensor = torch.zeros_like(emb_list[0])
            #         emb_list += [pad_tensor] * mins
            #     emb_list = emb_list[:self.mutation_sampling_size]
            #     emb_tensor = torch.stack(emb_list)
            #     datanumeric.append(emb_tensor)
                
            for col in pd_sampling.columns:
                if col == "embedding" or col == "ref_embedding":
                    continue
                np_data = pd_sampling[col].to_numpy()
                if is_padding:
                    np_data = np.pad(np_data, (0, mins), mode='constant', constant_values=0)
                np_data = np.asarray(np_data[:self.mutation_sampling_size], dtype=int)
                datanumeric.append(torch.tensor(np_data, dtype=torch.long))
            # Ensure datanumeric is valid
            datanumeric = torch.stack(datanumeric)
        # Ensure no None values in data_targets
        data_targets = {
            "class_index": idx_class if idx_class is not None else [],
            "subclass_index": idx_subclass if idx_subclass is not None else []
        }
        
        # add embedding (5000x1024) to 3 x 5000 datanumeric
        if self.get_embedding:
            embedding_tensor = torch.stack(pd_sampling['embedding'].tolist())
            ref_embedding_tensor = torch.stack(pd_sampling['ref_embedding'].tolist())
            if is_padding:
                pad_rows = self.mutation_sampling_size - embedding_tensor.shape[0]
                embedding_tensor = F.pad(embedding_tensor, pad=(0, 0, 0, pad_rows), mode='constant', value=0)
                
                pad_rows = self.mutation_sampling_size - ref_embedding_tensor.shape[0]
                ref_embedding_tensor = F.pad(ref_embedding_tensor, pad=(0, 0, 0, pad_rows), mode='constant', value=0)
            # print(embedding_tensor.shape, datanumeric.shape)
            datanumeric = torch.cat((datanumeric, embedding_tensor.T, ref_embedding_tensor.T), dim=0)
        return datanumeric, data_targets, sample_path




if __name__ == '__main__':
    from muat.model import ModelConfig
    from muat.util import LabelEncoderFromCSV, get_model
    def initialize_label_encoders(target_path, subtarget_path=None):
        target_handler = [LabelEncoderFromCSV(csv_file=target_path, class_name_col='class_name', class_index_col='class_index')]
        n_class = len(target_handler[0].classes_)

        n_subclass = None
        if subtarget_path is not None:
            le2 = LabelEncoderFromCSV(csv_file=subtarget_path, class_name_col='subclass_name', class_index_col='subclass_index')
            target_handler.append(le2)
            n_subclass = len(le2.classes_)

        return target_handler, n_class, n_subclass
    gfm_embedding = True
    
    arch = 'MuAtMotifPositionGESF_2Labels'
    extdir = 'muat/extfile'
    motif_path = f"{extdir}/dictMutation.tsv"
    pos_path =  f"{extdir}/dictChpos.tsv"
    ges_path = f"{extdir}/dictGES.tsv"
    dict_motif = pd.read_csv(motif_path, sep='\t')
    dict_pos = pd.read_csv(pos_path, sep='\t')
    dict_ges = pd.read_csv(ges_path, sep='\t')
    mutation_type = "snv+mnv+indel"
    model_config = ModelConfig(
        model_name=arch,
        dict_motif=dict_motif,
        dict_pos=dict_pos,
        dict_ges=dict_ges,
        mutation_sampling_size=5000,
        n_layer=2,
        n_emb=128,
        n_head=2,
        n_class=None,
        mutation_type=mutation_type,
        gfm_embedding=gfm_embedding
    )    
    save_label_1 = "data/label_1.tsv"
    save_label_2 = "data/label_2.tsv"
    target_handler, n_class, n_subclass = initialize_label_encoders(save_label_1, save_label_2)
    model_config.num_class = n_class
    if n_subclass is not None:
        model_config.num_subclass = n_subclass

    # trainer_config.target_handler = target_handler

    model = get_model(arch, model_config)

    train_dataloader_config = DataloaderConfig(
            model_input={'motif': True,
                'pos': True,
                'ges': True
                },
            mutation_type_ratio={'snv': 0.4,'mnv': 0.4,'indel': 0.2,'sv_mei': 0,'neg': 0},
            mutation_sampling_size=5000,
            sampling_replacement=False,
    )
    data = pd.read_csv("data/train_metadata.tsv", sep='\t', low_memory=False)
    dataloaderVal = MuAtDataloader(data, train_dataloader_config, get_embedding=True)
    print(f"len dataset: {len(dataloaderVal)}")
    
    #pdb.set_trace()
    # data, target, sample_path = dataloaderVal.__getitem__(974)
    # print(data,target)
    # print(data.shape)
    print("Loading dataloader")
    dataloader= torch.utils.data.DataLoader(
            dataloaderVal, 
            batch_size=4, 
            # shuffle=True, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
    )
    print("Starting inference dataloader")
    idx = 0
    with torch.set_grad_enabled(False):
        for data, target, p in dataloader:
            print(f"Iter: {idx}")
            logits, _ = model(data)
            idx += 1

    
    # pdb.set_trace()



    # for k in range(0,len(dataloaderVal)):
    #     print(k)
    #     data,target,sample_path = dataloaderVal.__getitem__(k)
    #     print(data,target)

    exit(0)

    #dataloader = PCAWG(dataset_name = 'PCAWG', data_dir='/csc/epitkane/projects/PCAWG/shuffled_samples/', mode='training',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True)

    #dataloader = PCAWG(dataset_name = 'pcawg_mut3_comb0', data_dir='/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/all24classes/', mode='training',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=3,addposition=False,filter=False,topk=5000)
    #dataloaderVal = PCAWG(dataset_name = 'pcawg_mut3_comb0', data_dir='/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/all24classes/', mode='validation',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=3,addposition=False,filter=False,topk=5000)
    #/csc/epitkane/projects/tcga/new23classes/
    #/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/new24classes/

    #G:/experiment/data/new24classes/
    '''
    dataloaderVal = FinalTCGAPCAWG(dataset_name = 'finalpcawg', 
                                data_dir='G:/experiment/data/new24classes/', 
                                mode='validation', 
                                curr_fold=1, 
                                block_size=5000, 
                                load=False,
                                mutratio = '0.3-0.3-0.3-0-0',
                                addtriplettoken=False,
                                addpostoken=False,
                                addgestoken=True,
                                addrt=False,
                                nummut = 0,
                                frac = 0,
                                adddatadir='G:/experiment/data/icgc/')

    #pdb.set_trace()
    data,target = dataloaderVal.__getitem__(0)
    pdb.set_trace()

    for k in range(0,len(dataloaderVal)):
        print(k)
        data,target = dataloaderVal.__getitem__(k)
    '''



    '''
    WGS GX
    '''

    #/scratch/project_2001668/data/pcawg

    dataloaderVal = TCGAPCAWG_Dataloader(dataset_name = 'wgsgx', 
                                        data_dir='/scratch/project_2001668/data/pcawg/allclasses/newformat/', 
                                        mode='training', 
                                        curr_fold=1, 
                                        block_size=5000, 
                                        load=False,
                                        addtriplettoken=True,
                                        addpostoken=False,
                                        addgestoken=False,
                                        addrt=False,
                                        nummut = 0,
                                        frac = 0,
                                        mutratio = '1-0-0-0-0',
                                        adddatadir = None,
                                        input_filename=None,
                                        args = None,
                                        gx_dir = '/scratch/project_2001668/data/pcawg/PCAWG_geneexp/')
    
    data,target = dataloaderVal.__getitem__(0)
    pdb.set_trace()

    '''
    fold = [1,2,3,4,5,6,7,8,9,10]
    mutratios = ['1-0-0-0-0','0.5-0.5-0-0-0','0.4-0.3-0.3-0-0','0.3-0.3-0.20-0.20-0','0.25-0.25-0.25-0.15-0.1']

    retrieve = ['addtriplettoken','addpostoken','addgestoken','addrt']

    for fo in fold:
        for i in retrieve:
            if i == 'addtriplettoken':
                addtriplettoken = True
            else:
                addtriplettoken = False
            
            if i == 'addpostoken':
                addpostoken = True
            else:
                addpostoken = False

            if i == 'addgestoken':
                addgestoken = True
            else:
                addgestoken = False

            if i == 'addrt':
                addrt = True
            else:
                addrt = False

            for j in mutratios:
                dataloaderVal = FinalTCGAPCAWG(dataset_name = 'finalpcawg', 
                                    data_dir='G:/experiment/data/new24classes/', 
                                    mode='validation', 
                                    curr_fold=1, 
                                    block_size=5000, 
                                    load=False,
                                    mutratio = j,
                                    addtriplettoken=addtriplettoken,
                                    addpostoken=addpostoken,
                                    addgestoken=addgestoken,
                                    addrt=addrt,
                                    nummut = 0,
                                    frac = 0)
                for k in range(0,len(dataloaderVal)):
                    print(str(fo) + ' ' + str(k) + ' ' + i + ' ' + j + ' ' + str(addtriplettoken) + str(addpostoken) + str(addgestoken) + str(addrt))
                    data,target = dataloaderVal.__getitem__(k)
    pdb.set_trace()

    dataloaderVal = TCGA(dataset_name = 'tcga_emb', data_dir='/csc/epitkane/projects/tcga/all23classes/', mode='validation',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=64,addposition=True,filter=True,block_size=300,withclass=True,twostream=False)

    for i in range(len(dataloaderVal)):
        data,target = dataloaderVal.__getitem__(i)

    dataloaderVal = TCGA(dataset_name = 'tcga_emb', data_dir='/csc/epitkane/projects/tcga/all23classes/', mode='testing',portion = [8,1,1], folds=10, curr_fold=1,load=True,load_token=True,ncontext=64,addposition=True,filter=True,block_size=300,loaddist=False,withclass=True,twostream=False)

    for i in range(len(dataloaderVal)):
        data,target = dataloaderVal.__getitem__(i)
    
    pdb.set_trace()
    '''
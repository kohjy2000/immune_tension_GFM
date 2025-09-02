import pandas as pd
import numpy as np
import glob
import os

# ───── CONFIGURE THESE PATHS ─────────────────────────────────────────────
METADATA_TSV = 'data/all_tumoursamples_pcawg.tsv'   
PREP_DIR      = 'data/inputs_preprocessed_pcawg'     # folder containing *.gz files
PREP_DIR      = 'data/inputs_preprocessed_pcawg_SinglePos'     # folder containing *.gz files
PREP_DIR      = 'data/inputs_preprocessed_pcawg_Norm_RefMut'     # folder containing *.gz files
SAVE_DIR      = 'data/'                         # where to write label_1.tsv, label_2.tsv, final.tsv
# ─────────────────────────────────────────────────────────────────────────
    
# 1) Load metadata and split tumour_type
meta = pd.read_csv(METADATA_TSV, sep='\t')

# meta = meta[~meta['submitted_sample_id'].str.contains('TCGA')].reset_index(drop=True)

splits = meta['tumour_type'].str.split('-', n=1)
print(splits.str[0])
meta['class_name']   = splits.str[0]
meta['subclass_name']= meta['tumour_type']

# 2) Build label_1 (class_index)
if 'class_index' in meta.columns:
    label_1 = (meta[['class_name','class_index']]
               .drop_duplicates()
               .sort_values('class_index')
               .reset_index(drop=True))
else:
    label_1 = (meta[['class_name']]
               .drop_duplicates()
               .sort_values('class_name')
               .reset_index(drop=True))
    label_1['class_index'] = np.arange(len(label_1))

label_1.to_csv(os.path.join(SAVE_DIR,'label_1.tsv'), sep='\t', index=False)

# 3) Build label_2 (subclass_index), if subclass_name exists
label_2 = None
if 'subclass_name' in meta.columns:
    if 'subclass_index' in meta.columns:
        label_2 = (meta[['subclass_name','subclass_index']]
                   .drop_duplicates()
                   .sort_values('subclass_index')
                   .reset_index(drop=True))
    else:
        label_2 = (meta[['subclass_name']]
                   .drop_duplicates()
                   .sort_values('subclass_name')
                   .reset_index(drop=True))
        label_2['subclass_index'] = np.arange(len(label_2))

    label_2.to_csv(os.path.join(SAVE_DIR,'label_2.tsv'), sep='\t', index=False)

# 4) Merge indices back into metadata
meta = meta.merge(label_1,      on='class_name',    how='left')
if label_2 is not None:
    meta = meta.merge(label_2,  on='subclass_name', how='left')

# 5) Gather prep files and extract sample IDs
files = glob.glob(os.path.join(PREP_DIR, '*muat.tsv'))
df_files = pd.DataFrame({
    'prep_path': files
})
# sample ID is the first token of the basename before the first “.”
df_files['sample'] = (
    df_files['prep_path']
      .apply(lambda p: os.path.basename(p).split('.')[0])
)

# sample ID is the first token of the basename before the first “.”
meta['sample'] = (
    meta['sample']
      .apply(lambda p: os.path.basename(p).split('.')[0])
)

# 6) Join with metadata and select final columns
print(df_files.head())
print(meta.head())
final = (
    df_files
      .merge(meta, on='sample', how='left')
    #   [['prep_path','class_name','subclass_name','class_index','subclass_index']]
)

# 6.1) Get embedding file paths and extract sample IDs
embedding_files = glob.glob(os.path.join(PREP_DIR, '*embedding.npz'))
df_emb = pd.DataFrame({'embedding_path': embedding_files})
df_emb['sample'] = df_emb['embedding_path'].apply(lambda p: os.path.basename(p).split('.')[0])
# open embeeding files and count number of rows in 'embeddings' column
df_emb['num_embeddings'] = df_emb['embedding_path'].apply(
    lambda p: np.load(p, allow_pickle=True)['meta'].shape[0]
)
# Filter out embedding files with less than 300 embeddings
df_emb = df_emb[300 <= df_emb['num_embeddings']]
df_emb = df_emb[df_emb['num_embeddings'] <= 20000].drop(columns='num_embeddings')

# Merge embedding paths into final dataframe
final = (
    final.merge(df_emb, on='sample', how='left')
    [['prep_path', 'embedding_path', 'class_name','subclass_name','class_index','subclass_index']]
)

# Drop those with embedding embedding path NaN
final = final[final['embedding_path'].notna()]

# # 7) Save out
# final.to_csv(os.path.join(SAVE_DIR,'final_metadata.tsv'), sep='\t', index=False)

# now split the data set to 9:1 to train and validation set.
# Do a stratified sampling based on the class_index + subclass_index
from sklearn.model_selection import train_test_split
train, val = train_test_split(
    final,
    test_size=0.15,
    stratify=final[['class_index','subclass_index']],
    # stratify=final[['class_index']],
    random_state=42
)
train.to_csv(os.path.join(SAVE_DIR,'train_metadata.tsv'), sep='\t', index=False)
val.to_csv(os.path.join(SAVE_DIR,'val_metadata.tsv'), sep='\t', index=False)

print("Wrote:\n • label_1.tsv\n • label_2.tsv\n • final_metadata.tsv\n • train_metadata.tsv\n • val_metadata.tsv")
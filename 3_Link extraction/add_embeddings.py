import pandas as pd

root_folder = '../2_Candidate generation/Data selection/'
# source_filenames = ['hypernym_candidates_tezaurs_20000_4', 'new_laws', 'synonym_candidates_top10000_4_meaning']
source_filenames = ['hypernym_candidates_tezaurs_20000_4', 'synonym_candidates_top10000_4_meaning']
# TODO - new_laws trūkst datu par sense_2 
# source_filenames = ['minitest']

dfs = []
for source_filename in source_filenames:
	filename = f'{root_folder}{source_filename}.csv'
	df = pd.read_csv(filename)
	df['source'] = source_filename
	print(df.columns.tolist())
	df = df.drop(columns=['sense1_gloss_embedding'], errors='ignore')
	df = df.drop(columns=['sense2_gloss_embedding'], errors='ignore')
	dfs.append(df)

# Find columns shared by all dataframes
shared_cols = set(dfs[0].columns)
for df in dfs[1:]:
    shared_cols &= set(df.columns)

print(f"Shared columns: {shared_cols}\n")

# Print columns that are NOT in the shared set for each dataframe
for i, df in enumerate(dfs, 1):
    extra_cols = set(df.columns) - shared_cols
    print(f"DataFrame {i} extra columns: {extra_cols}")

df = pd.concat(dfs, ignore_index=True)

embedding_df = pd.read_parquet("../1_Training process/Dataset/2_nouns_with_embeddings.parquet")
embedding_df = embedding_df[['sense_id', 'gloss_embedding', 'heading_embedding']]
print(df.columns.tolist())

df = df.merge(
    embedding_df.rename(columns={
        'sense_id': 'sense1_id',
        'gloss_embedding': 'sense1_gloss_embedding',
        'heading_embedding': 'sense1_heading_embedding'
    }),
    on='sense1_id',
    how='left'
)
print(df.columns.tolist())

df = df.merge(
    embedding_df.rename(columns={
        'sense_id': 'sense2_id',
        'gloss_embedding': 'sense2_gloss_embedding',
        'heading_embedding': 'sense2_heading_embedding'
    }),
    on='sense2_id',
    how='left'
)
print(df.columns.tolist())
df = df.drop_duplicates(subset=['sense1_id', 'sense2_id'], keep='first')

# Print rows where any embedding could not be found
missing = df[
    df[['sense1_gloss_embedding', 'sense1_heading_embedding']].isna().any(axis=1)
]
print(f"Rows with missing embeddings ({len(missing)} rows):")
print(missing)

# Print rows where any embedding could not be found
missing = df[
    df[['sense2_gloss_embedding', 'sense2_heading_embedding']].isna().any(axis=1)
][['sense2_id', 'sense2_heading', 'sense2_gloss']]
print(f"Rows with missing embeddings ({len(missing)} rows):")
print(missing)

# Drop rows where any of the embedding columns are missing
df = df.dropna(subset=[
    'sense1_gloss_embedding', 'sense2_gloss_embedding',
    'sense1_heading_embedding', 'sense2_heading_embedding'
])
df.to_parquet("candidates_with_embeddings.parquet",index=False, compression="zstd")

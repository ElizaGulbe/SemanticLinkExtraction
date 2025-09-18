import pandas as pd
import torch
import tqdm
import pickle  # Import pickle for serialization
import embeddings.hplt_embedding as embedding  # Change import if you want to use a different embedding
from transformers import AutoTokenizer, AutoModel
import base64

"""
The purpose of this file is to embed the dataset (headings and its respective senses). 
These embeddings will be used later in the training process. 
"""
"""The input file is the product of 1_Training process/Dataset/1_positive_example_extraction.py csv file. 
Each row of the input file contains two words senses and the relationship recorded between them alongside with ID parameters. The purpose of this file
is to compute embeddings for the respective headings and senses. These embedding will be used in the model training process.""" 

"""Currently we are using HPLT embeddings for the Latvian language. You can add new embeddings under 1_Training process/Dataset/embeddings """

def serialize_tensor(tensor):
    return base64.b64encode(tensor.cpu().numpy().tobytes()).decode("utf-8")

def deserialize_tensor(s, dtype=torch.float32, shape=(1, 768)):
    arr = np.frombuffer(base64.b64decode(s), dtype=dtype)
    return torch.from_numpy(arr).reshape(shape)

tokenizer = AutoTokenizer.from_pretrained("HPLT/hplt_bert_base_lv")
model = AutoModel.from_pretrained("HPLT/hplt_bert_base_lv", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the GPU if available
model.to(device)

df = pd.read_parquet("1_positive_examples_nouns.parquet")
df["sense1_gloss_embedding"] = None
df["sense2_gloss_embedding"] = None
df["sense1_heading_embedding"] = None
df["sense2_heading_embedding"] = None
df_nouns = pd.read_parquet("1_nouns.parquet")
df_nouns["gloss_embedding"] = None
df_nouns["heading_embedding"] = None

# # TESTTESTTESST
# # Keep only the first 3 rows
# df_nouns = df_nouns.head(3).copy()
# df_nouns["gloss"] = ["zaķis", "meitene", "kaķis"]
# df_nouns["entry_heading"] = ["zaķis", "zaķis", "kaķis"]
# print(df_nouns)
# print('zaķis',serialize_tensor(embedding.get_embedding('zaķis', tokenizer, model, device))[:100])
# print('meitene', serialize_tensor(embedding.get_embedding('meitene', tokenizer, model, device))[:100])
# print('kaķis', serialize_tensor(embedding.get_embedding('kaķis', tokenizer, model, device))[:100])

# Update embeddings in the DataFrame 
"""
Because the embedding process takes a long time and we don't want to embed a specific row multiple times,
we embedd a heading / gloss only once, then find all instances of that heading / gloss and add the embedding
for all found instances. After the embeddings are computed we serialize them to string representation, because the csv format doesn't handle well vector formats. 
"""
for index, row in tqdm.tqdm(df_nouns.iterrows(), total=df_nouns.shape[0]):
    if pd.isna(row['gloss_embedding']):
        text_gloss = str(row['gloss'])
        embedding_gloss = serialize_tensor(embedding.get_embedding(text_gloss, tokenizer, model, device))
        df_nouns.loc[df_nouns['sense_id'] == row["sense_id"], 'gloss_embedding'] = embedding_gloss
        df.loc[df['sense1_id'] == row["sense_id"], 'sense1_gloss_embedding'] = embedding_gloss
        df.loc[df['sense2_id'] == row["sense_id"], 'sense2_gloss_embedding'] = embedding_gloss
    if pd.isna(row['heading_embedding']):
        text_heading = str(row['entry_heading'])
        embedding_heading = serialize_tensor(embedding.get_embedding(text_heading, tokenizer, model, device))
        df_nouns.loc[df_nouns['entry_id'] == row["entry_id"], 'heading_embedding'] = embedding_heading
        df.loc[df['sense1_id'] == row["sense_id"], 'sense1_heading_embedding'] = embedding_heading
        df.loc[df['sense2_id'] == row["sense_id"], 'sense2_heading_embedding'] = embedding_heading

# Save the DataFrame
df_nouns.to_parquet("2_nouns_with_embeddings.parquet", index=False, compression="zstd")

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    if pd.isna(row['sense1_gloss_embedding']):
        text_gloss_1 = str(row['sense1_gloss'])
        embedding_gloss_1 = serialize_tensor(embedding.get_embedding(text_gloss_1, tokenizer, model, device))
        df.loc[df['sense1_id'] == row["sense1_id"], 'sense1_gloss_embedding'] = embedding_gloss_1
        df.loc[df['sense2_id'] == row["sense1_id"], 'sense2_gloss_embedding'] = embedding_gloss_1
    if pd.isna(row['sense2_gloss_embedding']):
        text_gloss_2 = str(row['sense2_gloss'])
        embedding_gloss_2 = serialize_tensor(embedding.get_embedding(text_gloss_2, tokenizer, model, device))
        df.loc[df['sense1_id'] == row["sense2_id"], 'sense1_gloss_embedding'] = embedding_gloss_2
        df.loc[df['sense2_id'] == row["sense2_id"], 'sense2_gloss_embedding'] = embedding_gloss_2
    if pd.isna(row['sense1_heading_embedding']):
        text_heading_1 = str(row['sense1_heading'])
        embedding_heading_1 = serialize_tensor(embedding.get_embedding(text_heading_1, tokenizer, model, device))
        df.loc[df['sense1_id'] == row["sense1_id"], 'sense1_heading_embedding'] = embedding_heading_1
        df.loc[df['sense2_id'] == row["sense1_id"], 'sense2_heading_embedding'] = embedding_heading_1
    if pd.isna(row['sense2_gloss_embedding']):
        text_heading_2 = str(row['sense2_heading'])
        embedding_heading_2 = serialize_tensor(embedding.get_embedding(text_heading_2, tokenizer, model, device))
        df.loc[df['sense1_id'] == row["sense2_id"], 'sense1_heading_embedding'] = embedding_heading_2
        df.loc[df['sense2_id'] == row["sense2_id"], 'sense1_heading_embedding'] = embedding_heading_2
# Save the DataFrame
df.to_parquet("2_positive_examples_nouns_embedded.parquet", index=False, compression="zstd")

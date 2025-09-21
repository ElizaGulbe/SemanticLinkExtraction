import pandas as pd
import torch
from tqdm import tqdm
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
    if s is None or pd.isna(s):
        return None
    arr = np.frombuffer(base64.b64decode(s), dtype=np.float32).copy()
    return torch.from_numpy(arr).reshape(shape).to(dtype)

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

# Update embeddings in the DataFrame 
"""
Because the embedding process takes a long time and we don't want to embed a specific row multiple times,
we embedd a heading / gloss only once, then find all instances of that heading / gloss and add the embedding
for all found instances. After the embeddings are computed we serialize them to string representation, because the csv format doesn't handle well vector formats. 
"""
for idx in tqdm(df_nouns.index, total=df_nouns.shape[0]):
    if pd.isna(df_nouns.at[idx, 'gloss_embedding']):
        text_gloss = str(df_nouns.at[idx, 'gloss'])
        embedding_gloss = serialize_tensor(
            embedding.get_embedding(text_gloss, tokenizer, model, device)
        )
        sense_id = df_nouns.at[idx, 'sense_id']
        df_nouns.loc[df_nouns['sense_id'] == sense_id, 'gloss_embedding'] = embedding_gloss
        df.loc[df['sense1_id'] == sense_id, 'sense1_gloss_embedding'] = embedding_gloss
        df.loc[df['sense2_id'] == sense_id, 'sense2_gloss_embedding'] = embedding_gloss

    if pd.isna(df_nouns.at[idx, 'heading_embedding']):
        text_heading = str(df_nouns.at[idx, 'entry_heading'])
        embedding_heading = serialize_tensor(
            embedding.get_embedding(text_heading, tokenizer, model, device)
        )
        entry_id = df_nouns.at[idx, 'entry_id']
        sense_id = df_nouns.at[idx, 'sense_id']
        df_nouns.loc[df_nouns['entry_id'] == entry_id, 'heading_embedding'] = embedding_heading
        df.loc[df['sense1_id'] == sense_id, 'sense1_heading_embedding'] = embedding_heading
        df.loc[df['sense2_id'] == sense_id, 'sense2_heading_embedding'] = embedding_heading

# Save the DataFrame
df_nouns.to_parquet("2_nouns_with_embeddings.parquet", index=False, compression="zstd")

for idx in tqdm(df.index, total=df.shape[0]):
    # --- sense1 gloss ---
    if pd.isna(df.at[idx, 'sense1_gloss_embedding']):
        text_gloss_1 = str(df.at[idx, 'sense1_gloss'])
        embedding_gloss_1 = serialize_tensor(
            embedding.get_embedding(text_gloss_1, tokenizer, model, device)
        )
        sense1_id = df.at[idx, 'sense1_id']
        df.loc[df['sense1_id'] == sense1_id, 'sense1_gloss_embedding'] = embedding_gloss_1
        df.loc[df['sense2_id'] == sense1_id, 'sense2_gloss_embedding'] = embedding_gloss_1

    # --- sense2 gloss ---
    if pd.isna(df.at[idx, 'sense2_gloss_embedding']):
        text_gloss_2 = str(df.at[idx, 'sense2_gloss'])
        embedding_gloss_2 = serialize_tensor(
            embedding.get_embedding(text_gloss_2, tokenizer, model, device)
        )
        sense2_id = df.at[idx, 'sense2_id']
        df.loc[df['sense1_id'] == sense2_id, 'sense1_gloss_embedding'] = embedding_gloss_2
        df.loc[df['sense2_id'] == sense2_id, 'sense2_gloss_embedding'] = embedding_gloss_2

    # --- sense1 heading ---
    if pd.isna(df.at[idx, 'sense1_heading_embedding']):
        text_heading_1 = str(df.at[idx, 'sense1_heading'])
        embedding_heading_1 = serialize_tensor(
            embedding.get_embedding(text_heading_1, tokenizer, model, device)
        )
        sense1_id = df.at[idx, 'sense1_id']
        df.loc[df['sense1_id'] == sense1_id, 'sense1_heading_embedding'] = embedding_heading_1
        df.loc[df['sense2_id'] == sense1_id, 'sense2_heading_embedding'] = embedding_heading_1

    # --- sense2 heading ---
    if pd.isna(df.at[idx, 'sense2_heading_embedding']):
        text_heading_2 = str(df.at[idx, 'sense2_heading'])
        embedding_heading_2 = serialize_tensor(
            embedding.get_embedding(text_heading_2, tokenizer, model, device)
        )
        sense2_id = df.at[idx, 'sense2_id']
        df.loc[df['sense1_id'] == sense2_id, 'sense1_heading_embedding'] = embedding_heading_2
        df.loc[df['sense2_id'] == sense2_id, 'sense2_heading_embedding'] = embedding_heading_2
# Save the DataFrame
df.to_parquet("2_positive_examples_nouns_embedded.parquet", index=False, compression="zstd")

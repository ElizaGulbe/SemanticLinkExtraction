import pandas as pd
import torch
import tqdm
import pickle  # Import pickle for serialization
import embeddings.hplt_embedding as embedding  # Change import if you want to use a different embedding
from transformers import AutoTokenizer, AutoModel

"""
The purpose of this file is to embed the dataset. 
These embedding will be used later in the training process. 
"""

tokenizer = AutoTokenizer.from_pretrained("HPLT/hplt_bert_base_lv")
model = AutoModel.from_pretrained("HPLT/hplt_bert_base_lv", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the GPU if available
model.to(device)

df = pd.read_csv("Training process/Dataset/positive_examples_nouns.csv")
df["sense1_gloss_embedding"] = None
df["sense2_gloss_embedding"] = None
df["sense1_heading_embedding"] = None
df["sense2_heading_embedding"] = None

# Function to serialize tensor to string representation
def serialize_tensor(tensor):
    return pickle.dumps(tensor).hex()

# Update embeddings in the DataFrame 
"""
Because the embedding process takes a long time and we don't want to embed a specific row multiple times,
we embedd a heading / gloss only once, then find all instances of that heading / gloss and add the embedding
for all found instances
"""
for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    if pd.isna(row['sense1_gloss_embedding']):
        text_gloss_1 = str(row['sense1_gloss'])
        embedding_gloss_1 = embedding.get_embedding(text_gloss_1, tokenizer, model, device)  # Replace with actual embedding logic
        df.loc[df['sense1_id'] == row["sense1_id"], 'sense1_gloss_embedding'] = serialize_tensor(embedding_gloss_1)
        df.loc[df['sense2_id'] == row["sense1_id"], 'sense2_gloss_embedding'] = serialize_tensor(embedding_gloss_1)
    if pd.isna(row['sense2_gloss_embedding']):
        text_gloss_2 = str(row['sense2_gloss'])
        embedding_gloss_2 = embedding.get_embedding(text_gloss_2, tokenizer, model, device)  # Replace with actual embedding logic
        df.loc[df['sense1_id'] == row["sense2_id"], 'sense1_gloss_embedding'] = serialize_tensor(embedding_gloss_2)
        df.loc[df['sense2_id'] == row["sense2_id"], 'sense2_gloss_embedding'] = serialize_tensor(embedding_gloss_2)
    if pd.isna(row['sense1_heading_embedding']):
        text_heading_1 = str(row['sense1_heading'])
        embedding_heading_1 = embedding.get_embedding(text_heading_1, tokenizer, model, device)  # Replace with actual embedding logic
        df.loc[df['sense1_id'] == row["sense1_id"], 'sense1_heading_embedding'] = serialize_tensor(embedding_heading_1)
        df.loc[df['sense2_id'] == row["sense1_id"], 'sense2_heading_embedding'] = serialize_tensor(embedding_heading_1)
    if pd.isna(row['sense2_gloss_embedding']):
        text_heading_2 = str(row['sense2_heading'])
        embedding_heading_2 = embedding.get_embedding(text_heading_2, tokenizer, model, device)  # Replace with actual embedding logic
        df.loc[df['sense1_id'] == row["sense2_id"], 'sense1_heading_embedding'] = serialize_tensor(embedding_heading_2)
        df.loc[df['sense2_id'] == row["sense2_id"], 'sense1_heading_embedding'] = serialize_tensor(embedding_heading_2)


# Save the DataFrame to a CSV file
df.to_csv("Production/Prepare dataset/nouns_relation_embeddings.csv", index=False)
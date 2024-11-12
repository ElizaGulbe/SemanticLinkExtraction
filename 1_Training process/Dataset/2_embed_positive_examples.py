import pandas as pd
import torch
import tqdm
import pickle  # Import pickle for serialization
import embeddings.hplt_embedding as embedding  # Change import if you want to use a different embedding
from transformers import AutoTokenizer, AutoModel

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

# Function to deserialize string back into tensor
# sense1_id,sense1_entry_id,synset1_id,sense1_heading,sense1_gloss,heading1_PoS,sense2_id,sense2_entry_id,synset2_id,sense2_heading,sense2_gloss,heading2_PoS,rel_type
# Update embeddings in the DataFrame
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
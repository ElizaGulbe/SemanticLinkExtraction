import pandas as pd 
from tqdm import tqdm 
import pickle
import base64
tqdm.pandas(desc="Generating random examples")
import numpy as np
import torch

"""In every model training process it is important to add negative examples. Because of the size of the dataset, I focused only on 
hypernymy and synonymy because there was insufficient training data for other examples. """

"""This code covers several strategies for generating negative examples. 

1. Higher-level hypernyms: those not falling under direct hypernyms;
2. Random negative examples
3. Embedding vectors that have a close Euclidian distance but do not have an existing recorded relationship in the dataset
4. Unrelated senses of related words.
5. Similar/also/antonyms/holonyms - already recorded examples from different relation types that do not fall under the definition of hypernym or synonym  
"""


#df = pd.read_csv("Production/Prepare dataset/Source/nouns_relation_embeddings.csv")
#df_nouns = pd.read_csv("Production/Prepare dataset/Source/nouns_with_embeddings.csv")
# PP - (From another place) COuld this be the expected structure of df_nouns ?
# sense_id,entry_id,entry_heading,parent_sense_id,gloss,order_no,sense_tag,synset_id,hidden,gloss_embedding,heading_embedding,occuranceRatio
df = pd.read_parquet("2_positive_examples_nouns_embedded.parquet")
df_nouns = pd.read_parquet("2_nouns_with_embeddings.parquet")


def compute_top_euclidean_similarities_np(np_tensor, top_n=5):
    # Step 1: Compute the squared sum per row (norm squared)
    squared_tensor = np.sum(np.square(np_tensor), axis=1, keepdims=True)
    
    # Compute the full squared Euclidean distance matrix
    euclidean_distance_matrix = squared_tensor + squared_tensor.T - 2 * (np_tensor.dot(np_tensor.T))
    
    # Step 2: Take the square root to get actual Euclidean distances
    # Using np.maximum to prevent negative values due to numerical precision issues
    euclidean_distance_matrix = np.sqrt(np.maximum(euclidean_distance_matrix, 0.0))
    
    # Step 3: Iterate over each row and get top N smallest distances (ignoring self-distance)
    top_indices_list = []
    for i in tqdm(range(euclidean_distance_matrix.shape[0])):
        # Copy the row to avoid modifying the original matrix
        distances = np.copy(euclidean_distance_matrix[i])
        
        # Set the self-distance to infinity so it won't be selected as a nearest neighbor
        distances[i] = np.inf
        
        # Get the indices of the top N smallest distances
        top_indices = np.argsort(distances)[:top_n]
        top_indices_list.append(top_indices)
    
    return top_indices_list

def serialize_tensor(tensor):
    return base64.b64encode(tensor.cpu().numpy().tobytes()).decode("utf-8")

def deserialize_tensor(s, dtype=torch.float32, shape=(1, 768)):
    if s is None or pd.isna(s):
        return None
    arr = np.frombuffer(base64.b64decode(s), dtype=np.float32).copy()
    return torch.from_numpy(arr).reshape(shape).to(dtype)

embeddings_only = []
def add_embedding_only_row(row):
    new_row_1 = {
    "sense_id": row["sense1_id"],
    "sense_entry_id": row["sense1_entry_id"],
    "synset_id": row["synset1_id"],
    "sense_heading" : row["sense1_heading"],
    "sense_gloss" : row["sense1_gloss"],
    "heading_PoS" : row["heading1_PoS"],
    "sense_gloss_embedding" : deserialize_tensor(row["sense1_gloss_embedding"]),
    'heading_embedding': row['sense1_heading_embedding']
}
    new_row_2 = {
    "sense_id": row["sense2_id"],
    "sense_entry_id": row["sense2_entry_id"],
    "synset_id": row["synset2_id"],
    "sense_heading" : row["sense2_heading"],
    "sense_gloss" : row["sense2_gloss"],
    "heading_PoS" : row["heading2_PoS"],
    "sense_gloss_embedding" : deserialize_tensor(row["sense2_gloss_embedding"]),
    'heading_embedding': row['sense2_heading_embedding']
}
    embeddings_only.append(new_row_1)
    embeddings_only.append(new_row_2)

none_relations_similarity = []
def add_none_similarity_relation(row1, row2): 
    new_row = {
    'sense1_id':row1["sense_id"],
    'sense1_entry_id': row1["sense_entry_id"],
    'synset1_id': row1["synset_id"],
    'sense1_heading': row1['sense_heading'],
    'sense1_gloss': row1['sense_gloss'],
    'heading1_PoS': row1['heading_PoS'],
    'sense2_id': row2["sense_id"],
    'sense2_entry_id': row2["sense_entry_id"],
    'synset2_id': row2["synset_id"],
    'sense2_heading':row2['sense_heading'],
    'sense2_gloss': row2['sense_gloss'],
    'heading2_PoS': row2['heading_PoS'],
    'sense1_gloss_embedding': serialize_tensor(row1["sense_gloss_embedding"]),
    'sense2_gloss_embedding' : serialize_tensor(row2["sense_gloss_embedding"]),
    'sense1_heading_embedding': row1["heading_embedding"],
    'sense2_heading_embedding' : row2["heading_embedding"],
    'rel_type': "none_similarity"
}
    none_relations_similarity.append(new_row)

none_relations_random = []
def add_new_row_random(row1, row2):
    new_row = {
    'sense1_id': row1["sense1_id"],
    'sense1_entry_id': row1["sense1_entry_id"],
    'synset1_id': row1["synset1_id"],
    'sense1_heading': row1['sense1_heading'],
    'sense1_gloss': row1['sense1_gloss'],
    'heading1_PoS': row1['heading1_PoS'],
    'sense2_id': row2["sense2_id"],
    'sense2_entry_id': row2["sense2_entry_id"],
    'synset2_id': row2["synset2_id"],
    'sense2_heading': row2['sense2_heading'],
    'sense2_gloss': row2['sense2_gloss'],
    'heading2_PoS': row2['heading2_PoS'],
    'rel_type': "none_random",
    'sense1_gloss_embedding': row1["sense1_gloss_embedding"],
    'sense2_gloss_embedding' :row2["sense2_gloss_embedding"],
    'sense1_heading_embedding': row1["sense1_heading_embedding"],
    'sense2_heading_embedding' : row2["sense2_heading_embedding"],
}
    none_relations_random.append(new_row)

none_relations_grandparents = []
def add_new_row_grandparents(row1, row2):
    new_row = {
    'sense1_id': row1["sense1_id"],
    'sense1_entry_id': row1["sense1_entry_id"],
    'synset1_id': row1["synset1_id"],
    'sense1_heading': row1['sense1_heading'],
    'sense1_gloss': row1[ 'sense1_gloss'],
    'heading1_PoS': row1[ 'heading1_PoS'],
    'sense2_id': row2["sense2_id"],
    'sense2_entry_id': row2["sense2_entry_id"],
    'synset2_id': row2["synset2_id"],
    'sense2_heading': row2['sense2_heading'],
    'sense2_gloss': row2[ 'sense2_gloss'],
    'heading2_PoS': row2[ 'heading2_PoS'],
    'rel_type': "none_grandparents",
    'sense1_gloss_embedding': row1["sense1_gloss_embedding"],
    'sense2_gloss_embedding' : row2["sense2_gloss_embedding"],
    'sense1_heading_embedding': row1["sense1_heading_embedding"],
    'sense2_heading_embedding' : row2["sense2_heading_embedding"],
}
    none_relations_grandparents.append(new_row)


# get all unique senses with their embeddings 
# TO DO - just use only embeddings file. 
for index, row in tqdm(df.iterrows(),total=df.shape[0]):
    add_embedding_only_row(row)
df_individual_senses_emb = pd.DataFrame(embeddings_only)
df_individual_senses_emb = df_individual_senses_emb.drop_duplicates(subset=['sense_id'],keep='first')
df_individual_senses_emb = df_individual_senses_emb.reset_index(drop=True)
tensor_list = df_individual_senses_emb['sense_gloss_embedding'].tolist()
tensor_list_reduce_dim = [item[0] for item in tensor_list] 
np_tensor = np.stack(tensor_list_reduce_dim)


def generate_close_distance_candidates():
    top_similarities_indices = compute_top_euclidean_similarities_np(np_tensor, top_n=10)
    for i, indices in tqdm(enumerate(top_similarities_indices)):
        sense_id = df_individual_senses_emb.loc[i,"sense_id"]
        all_instances_sense_id = df[(df["sense1_id"] == sense_id) | (df["sense2_id"] == sense_id)]
        for idx in indices: 
            candidate_sense_id = df_individual_senses_emb.loc[idx,"sense_id"]
            relation_exists = (candidate_sense_id in all_instances_sense_id['sense1_id'] ) | (candidate_sense_id in all_instances_sense_id['sense2_id'])
            if not relation_exists: 
                add_none_similarity_relation(df_individual_senses_emb.loc[i,:],df_individual_senses_emb.loc[int(idx),:])


def generate_random_negative_examples(row):
    synset_id = row["synset1_id"]
    filtered_rows = df[(df["synset1_id"] != synset_id) & (df["synset2_id"] != synset_id)]
    sample_row = filtered_rows.sample(n=1).iloc[0]
    add_new_row_random(row,sample_row)

def find_hypernym_grandparanets():
    df_hypernymy = df[df['rel_type'] == 'hypernym']
    for idx, row in tqdm(df_hypernymy.iterrows(), total=df_hypernymy.shape[0],desc="Finding grandparents"):
        synset1_id = row["synset1_id"]
        grandparents_df = df_hypernymy[df_hypernymy["synset2_id"] == synset1_id]
        if grandparents_df.empty:
            continue
        else:
            add_new_row_grandparents(grandparents_df.sample(n=1).iloc[0,:],row)


asymetric_relationships = []
def add_asymetric_row(row,rel_type):
    new_row = {
    'sense1_id': row["sense2_id"],
    'sense1_entry_id': row["sense2_entry_id"],
    'synset1_id': row["synset2_id"],
    'sense1_heading': row['sense2_heading'],
    'sense1_gloss': row['sense2_gloss'],
    'heading1_PoS': row['heading2_PoS'],
    'sense2_id': row["sense1_id"],
    'sense2_entry_id': row["sense1_entry_id"],
    'synset2_id': row["synset1_id"],
    'sense2_heading': row['sense1_heading'],
    'sense2_gloss': row['sense1_gloss'],
    'heading2_PoS': row['heading1_PoS'],
    'sense1_gloss_embedding': row["sense2_gloss_embedding"],
    'sense2_gloss_embedding' : row["sense1_gloss_embedding"],
    'sense1_heading_embedding': row["sense2_heading_embedding"],
    'sense2_heading_embedding' : row["sense1_heading_embedding"],
    'rel_type': rel_type
}
    asymetric_relationships.append(new_row)

gloss_none = []
def add_gloss_none_row_hypernym(row,row_hyp_none):
    new_row = {
        'sense1_id': row["sense1_id"],
        'sense1_entry_id': row["sense1_entry_id"],
        'synset1_id': row["synset1_id"],
        'sense1_heading': row['sense1_heading'],
        'sense1_gloss': row['sense1_gloss'],
        'heading1_PoS': row['heading1_PoS'],
        'sense2_id': row_hyp_none["sense_id"],
        'sense2_entry_id': row_hyp_none["entry_id"],
        'synset2_id': row_hyp_none["synset_id"],
        'sense2_heading': row_hyp_none['entry_heading'],
        'sense2_gloss': row_hyp_none['gloss'],
        'heading2_PoS': "Lietvārds",
        'sense1_gloss_embedding': row["sense1_gloss_embedding"],
        'sense2_gloss_embedding' : row_hyp_none["gloss_embedding"],
        'sense1_heading_embedding' : row["sense1_heading_embedding"],
        'sense2_heading_embedding': row_hyp_none["heading_embedding"],
        'rel_type': "none_gloss_hypernym"
    }
    gloss_none.append(new_row)

def add_gloss_none_row_hyponym(row,row_hyp_none):
    new_row = {
        'sense2_id': row["sense1_id"],
        'sense2_entry_id': row["sense1_entry_id"],
        'synset2_id': row["synset1_id"],
        'sense2_heading': row['sense1_heading'],
        'sense2_gloss': row['sense1_gloss'],
        'heading2_PoS': row['heading1_PoS'],
        'sense1_id': row_hyp_none["sense_id"],
        'sense1_entry_id': row_hyp_none["entry_id"],
        'synset1_id': row_hyp_none["synset_id"],
        'sense1_heading': row_hyp_none['entry_heading'],
        'sense1_gloss': row_hyp_none['gloss'],
        'heading1_PoS': "Lietvārds",
        'sense1_gloss_embedding' : row_hyp_none["gloss_embedding"],
        'sense2_gloss_embedding': row["sense1_gloss_embedding"],
        'sense1_heading_embedding' : row_hyp_none["heading_embedding"],
        'sense2_heading_embedding': row["sense1_heading_embedding"],
        'rel_type': "none_gloss_hyponym"
    }
    gloss_none.append(new_row)

# sense_id,entry_id,entry_heading,parent_sense_id,gloss,order_no,sense_tag,synset_id,hidden,gloss_embedding,heading_embedding,occuranceRatio
def add_gloss_none_row_synonym(row1,row2):
    new_row = {
        'sense1_id': row1["sense_id"],
        'sense1_entry_id': row1["entry_id"],
        'synset1_id': row1["synset_id"],
        'sense1_heading': row1['entry_heading'],
        'sense1_gloss': row1['gloss'],
        'heading1_PoS': "Lietvārds",
        'sense2_id': row2["sense_id"],
        'sense2_entry_id': row2["entry_id"],
        'synset2_id': row2["synset_id"],
        'sense2_heading': row2['entry_heading'],
        'sense2_gloss': row2['gloss'],
        'heading2_PoS': "Lietvārds",
        'sense1_gloss_embedding': row1["gloss_embedding"],
        'sense2_gloss_embedding': row2["gloss_embedding"],
        'sense1_heading_embedding': row1["heading_embedding"],
        'sense2_heading_embedding': row2["heading_embedding"],
        'rel_type': "none_gloss_synonym"
    }
    gloss_none.append(new_row)

def add_asymetric_relationships():
    df_hypernymy = df[df['rel_type'] == 'hypernym']
    for idx, row in tqdm(df_hypernymy.iterrows(),total=df_hypernymy.shape[0],desc="Adding asymetric relations -hyponymy"):
        add_asymetric_row(row,"hyponym")
    df_holonym =  df[df['rel_type'] == 'holonym']
    for idx, row in tqdm(df_holonym.iterrows(),total=df_holonym.shape[0],desc="Adding asymetric relations - holonomy"):
        add_asymetric_row(row,"meronym")

# sense_id,entry_id,entry_heading,parent_sense_id,gloss,order_no,sense_tag,synset_id,hidden,gloss_embedding,heading_embedding,occuranceRatio

def add_gloss_none_relationship():
    df_hypernymy = df[df['rel_type'] == 'hypernym']
    for idx, row in tqdm(df_hypernymy.iterrows(),total=df_hypernymy.shape[0],desc="Adding from gloss -hypernymy"):
        hypernym_sense_id = row["sense2_id"]
        hypernym_heading_id = row["sense2_entry_id"]
        gloss = df_nouns[(df_nouns["entry_id"] == hypernym_heading_id) & (df_nouns["sense_id"] != hypernym_sense_id)]
        for _ , row_hyp_none in gloss.iterrows():
            add_gloss_none_row_hypernym(row,row_hyp_none)
            add_gloss_none_row_hyponym(row,row_hyp_none)

    df_synonymy = df[df['rel_type'] == 'synonymy']
    for idx, row in tqdm(df_synonymy.iterrows(),total=df_synonymy.shape[0],desc="Adding from gloss -synonymy"):        
        sense_tag_1 = df_nouns[df_nouns["sense_id"] == row['sense1_id']]["sense_tag"].iloc[0]
        sense_tag_2 = df_nouns[df_nouns["sense_id"] == row['sense2_id']]["sense_tag"].iloc[0] # get if sense tag is a,b,c ..

        # Check if the value is a string before accessing [0]
        if pd.notna(sense_tag_1) and isinstance(sense_tag_1, str):
            sense_tag_1 = sense_tag_1[0]  # Get the first character if it's a string
        else:
            continue
        if pd.notna(sense_tag_2) and isinstance(sense_tag_1, str):
            sense_tag_2 = sense_tag_2[0]  # Get the first character if it's a string
        else:
            continue

        heading_1 = row["sense1_entry_id"]
        heading_2 = row["sense2_entry_id"]
        if heading_1 == heading_2:
            continue
        gloss_heading_1 = df_nouns[(df_nouns["entry_id"] == heading_1) & (~df_nouns['sense_tag'].fillna('').str.contains(sense_tag_1, case=False))]
        gloss_heading_2 = df_nouns[(df_nouns["entry_id"] == heading_2) & (~df_nouns['sense_tag'].fillna('').str.contains(sense_tag_2, case=False))]
        for _ , entry1 in gloss_heading_1.iterrows():
            for _, entry2 in gloss_heading_2.iterrows():
                add_gloss_none_row_synonym(entry1,entry2)
                add_gloss_none_row_synonym(entry2,entry1)
 

for idx, row in tqdm(df.iterrows(),total=df.shape[0]): # samples random rows 
    generate_random_negative_examples(row)
find_hypernym_grandparanets()
#add_asymetric_relationships() # you can uncomment this if you want.
"""
# add_asymetric_relationships () switches sense_1 and sense_2 in place to add a new relationship hyponym / holonym, however, it is not used at the moment
# 
"""
add_gloss_none_relationship()
# For example if there is heading A with gloss containing sense 1, sense 2, sense 3, and heading B with gloss containing 
# sense 4, sense 5, and sense 6. We have recorded that sense 2 from heading A and sense 4 from heading B are synonyms. Therefore, 
# we create negative example by taking, for example, sense 1 and sense 5 
generate_close_distance_candidates()
none_grandparents_relations_df = pd.DataFrame(none_relations_grandparents)
none_random_relations_df = pd.DataFrame(none_relations_random)            
none_similarity_relations_df = pd.DataFrame(none_relations_similarity)
asymetric_relationships_df = pd.DataFrame(asymetric_relationships)
none_gloss = pd.DataFrame(gloss_none)
df = pd.concat([df,none_similarity_relations_df,none_random_relations_df,none_grandparents_relations_df,asymetric_relationships_df,none_gloss],axis=0,ignore_index=True)
df = df.drop_duplicates(subset=["sense1_id","sense2_id"], keep="first")
print(df['rel_type'].value_counts())
print("sense1_gloss_embedding trūkst", df["sense1_gloss_embedding"].isna().sum())
print("sense2_gloss_embedding trūkst", df["sense2_gloss_embedding"].isna().sum())
print("sense1_heading_embedding trūkst", df["sense1_heading_embedding"].isna().sum())
print("sense2_heading_embedding trūkst", df["sense2_heading_embedding"].isna().sum())
pd.crosstab(df["rel_type"], df["sense1_gloss_embedding"].isna())
pd.crosstab(df["rel_type"], df["sense2_gloss_embedding"].isna())
df = df.dropna(subset=["sense1_gloss_embedding", "sense2_gloss_embedding", "sense1_heading_embedding", "sense2_heading_embedding"])
print(df['rel_type'].value_counts())
df.to_parquet("training_dataset_nouns_pp.parquet",index=False, compression="zstd")

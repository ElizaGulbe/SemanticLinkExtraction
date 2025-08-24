import pandas as pd 
from tqdm import tqdm 
import tensorflow as tf
import pickle
tqdm.pandas(desc="Generating random examples")
# sense1_id,sense1_entry_id,synset1_id,sense1_heading,sense1_gloss,heading1_PoS,sense2_id,sense2_entry_id,synset2_id,sense2_heading,sense2_gloss,heading2_PoS,rel_type,sense1_gloss_embedding,sense2_gloss_embedding

import numpy as np

# Assuming tf_tensor is your TensorFlow tensor containing all embeddings
# tf_tensor shape is expected to be [num_tensors, embedding_size]

def compute_top_euclidean_similarities(tf_tensor, top_n=5):
    # Step 1: Compute the Euclidean distance matrix
    squared_tensor = tf.reduce_sum(tf.square(tf_tensor), axis=1, keepdims=True)
    euclidean_distance_matrix = squared_tensor + tf.transpose(squared_tensor) - 2 * tf.matmul(tf_tensor, tf.transpose(tf_tensor))

    # Step 2: Take the square root to get actual Euclidean distances
    euclidean_distance_matrix = tf.sqrt(tf.maximum(euclidean_distance_matrix, 0.0))  # ensure no negative values

    # Step 3: Iterate through each tensor and get top N smallest distances
    top_indices_list = []

    # Use tqdm to show progress during the iteration
    for i in tqdm(range(euclidean_distance_matrix.shape[0])):
        # Get the Euclidean distances for tensor i with all others
        distances = euclidean_distance_matrix[i]

        # Set the self-distance to +inf to exclude it from being selected
        distances = tf.tensor_scatter_nd_update(
            distances,
            indices=[[i]],
            updates=[np.inf]
        )

        # Get the top N indices with the smallest distances (i.e., most similar)
        top_indices = tf.argsort(distances, direction='ASCENDING')[:top_n]

        top_indices_list.append(top_indices.numpy())

    return top_indices_list




def deserialize_tensor(serialized_tensor):
    return pickle.loads(bytes.fromhex(serialized_tensor))

def serialize_tensor(tensor):
    return pickle.dumps(tensor).hex()


# /
df = pd.read_csv("Production/Prepare dataset/Source/nouns_relation_embeddings.csv")
df_nouns = pd.read_csv("Production/Prepare dataset/Source/nouns_with_embeddings.csv")
i=0

embeddings_only = []
def add_embedding_only_row(row):
    new_row_1 = {
    "sense_id": row["sense1_id"],
    "sense_entry_id": row["sense1_entry_id"],
    "synset_id": row["synset1_id"],
    "sense_heading" : row["sense1_heading"],
    "sense_gloss" : row["sense1_gloss"],
    "heading_PoS" : row["heading1_PoS"],
    "sense_gloss_embedding" : deserialize_tensor(row["sense1_gloss_embedding"])
}
    new_row_2 = {
    "sense_id": row["sense2_id"],
    "sense_entry_id": row["sense2_entry_id"],
    "synset_id": row["synset2_id"],
    "sense_heading" : row["sense2_heading"],
    "sense_gloss" : row["sense2_gloss"],
    "heading_PoS" : row["heading2_PoS"],
    "sense_gloss_embedding" : deserialize_tensor(row["sense2_gloss_embedding"])
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
    'sense1_gloss': row1[ 'sense1_gloss'],
    'heading1_PoS': row1[ 'heading1_PoS'],
    'sense2_id': row2["sense2_id"],
    'sense2_entry_id': row2["sense2_entry_id"],
    'synset2_id': row2["synset2_id"],
    'sense2_heading': row2['sense2_heading'],
    'sense2_gloss': row2[ 'sense2_gloss'],
    'heading2_PoS': row2[ 'heading2_PoS'],
    'rel_type': "none_random",
    'sense1_gloss_embedding': row1["sense1_gloss_embedding"],
    'sense2_gloss_embedding' :row2["sense2_gloss_embedding"],
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
}
    none_relations_grandparents.append(new_row)


for index, row in tqdm(df.iterrows(),total=df.shape[0]):
    add_embedding_only_row(row)
df_only_embeddings = pd.DataFrame(embeddings_only)
df_only_embeddings = df_only_embeddings.drop_duplicates(subset=['sense_id'],keep='first')
df_only_embeddings = df_only_embeddings.reset_index(drop=True)
tensor_list = df_only_embeddings['sense_gloss_embedding'].tolist()
tensor_list_reduce_dim = [item[0] for item in tensor_list] 

tf_tensor = tf.stack(tensor_list_reduce_dim)

top_similarities_indices = compute_top_euclidean_similarities(tf_tensor, top_n=10)
for i, indices in tqdm(enumerate(top_similarities_indices)):
    sense_id = df_only_embeddings.loc[i,"sense_id"]
    all_instances_sense_id = df[(df["sense1_id"] == sense_id) | (df["sense2_id"] == sense_id)]
    for idx in indices: 
        candidate_sense_id = df_only_embeddings.loc[idx,"sense_id"]
        relation_exists = (candidate_sense_id in all_instances_sense_id['sense1_id'] ) | (candidate_sense_id in all_instances_sense_id['sense2_id'])
        if not relation_exists: 
            add_none_similarity_relation(df_only_embeddings.loc[i,:],df_only_embeddings.loc[int(idx),:])


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
    'sense2_gloss_embedding': row["sense1_gloss_embedding"],
    'sense1_gloss_embedding' : row_hyp_none["gloss_embedding"],
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
    'sense2_gloss_embedding' : row2["gloss_embedding"],
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
        if idx > 10000:
            break
        try:
            sense_tag_1 = df_nouns[df_nouns["sense_id"] == row['sense1_id']]["sense_tag"].iloc[0]
        except:
            continue
        # Check if the value is a string before accessing [0]
        if pd.notna(sense_tag_1) and isinstance(sense_tag_1, str):
            sense_tag_1 = sense_tag_1[0]  # Get the first character if it's a string
        else:
            continue
        try:
            sense_tag_2 =  df_nouns[df_nouns["sense_id"] == row['sense2_id']]["sense_tag"].iloc[0] # get if sense tag is a,b,c ..
        except:
            continue
        if pd.notna(sense_tag_2) and isinstance(sense_tag_1, str):
            sense_tag_2 = sense_tag_2[0]  # Get the first character if it's a string
        else:
            continue

        heading1 = row["sense1_entry_id"]
        heading2 = row["sense2_entry_id"]
        gloss_heading_1 = df_nouns[(df_nouns["entry_id"] == heading1) & (~df_nouns['sense_tag'].fillna('').str.contains(sense_tag_1, case=False))]

        gloss_heading_2 = df_nouns[(df_nouns["entry_id"] ==  heading2) & (~df_nouns['sense_tag'].fillna('').str.contains(sense_tag_2, case=False))]
        for _ , entry1 in gloss_heading_1.iterrows():
            for iddx, entry2 in  gloss_heading_2.iterrows():
                add_gloss_none_row_synonym(entry1,entry2)
                add_gloss_none_row_synonym(entry2,entry1)
 
        
        
for idx, row in tqdm(df.iterrows(),total=df.shape[0]):
    generate_random_negative_examples(row)

find_hypernym_grandparanets()
#add_asymetric_relationships()
add_gloss_none_relationship()
none_grandparents_relations_df = pd.DataFrame(none_relations_grandparents)
none_random_relations_df = pd.DataFrame(none_relations_random)            
none_similarity_relations_df = pd.DataFrame(none_relations_similarity)
asymetric_relationships_df = pd.DataFrame(asymetric_relationships)
none_gloss = pd.DataFrame(gloss_none)
df = pd.concat([df,none_similarity_relations_df,none_random_relations_df,none_grandparents_relations_df,asymetric_relationships_df,none_gloss],axis=0,ignore_index=True)
df = df.drop_duplicates(subset=["sense1_id","sense2_id"], keep="first")
df.to_csv("Production/training_dataset_nouns_with_none_hypernyms_synonyms_new2910.csv",index=False)
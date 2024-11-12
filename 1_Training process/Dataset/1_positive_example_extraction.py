import psycopg2
import pandas as pd
import itertools
from tqdm import tqdm
import os
from dotenv import load_dotenv

"""
The purpose of this file is to select the existing data about the synsets and its relation with other synsets. 
The outcome of this file is a CSV file containing unique relationships between word senses - synonymy, hypernymy, also, similar, antonym,meronym - 
which is later used as the basis for the training/test dataset. 
Currently we only work with nouns, but you can comment out this line "df = df[(df['heading1_PoS'] == 'Lietvārds') & (df['heading2_PoS'] == 'Lietvārds')]"
if you want to include other PoS in the training process. 

This code requires a connection to Tezaurs DB. 

"""
def get_synset_members(synset_id):
    synset_query = """
        SELECT senses.id, senses.entry_id, entries.heading, senses.gloss, paradigms.data
        FROM dict.senses
        JOIN dict.entries ON senses.entry_id = entries.id
        JOIN dict.lexemes ON senses.entry_id = lexemes.entry_id
        JOIN dict.paradigms ON lexemes.paradigm_id = paradigms.id
        WHERE senses.synset_id = %s;
    """
    cursor.execute(synset_query, (synset_id,))
    return cursor.fetchall()

        # Function to add row data
def create_row_data(row1, row2, synset1_id, synset2_id, rel_type):
    return {
        "sense1_id": row1[0],
        "sense1_entry_id": row1[1],
        "synset1_id": synset1_id,
        "sense1_heading": row1[2],
        "sense1_gloss": row1[3],
        "heading1_PoS": row1[4].get('Vārdšķira') if isinstance(row1[4], dict) else None,
        "sense2_id": row2[0],
        "sense2_entry_id": row2[1],
        "synset2_id": synset2_id,
        "sense2_heading": row2[2],
        "sense2_gloss": row2[3],
        "heading2_PoS": row2[4].get('Vārdšķira') if isinstance(row2[4], dict) else None,
        "rel_type": rel_type
    }

# Initialize the DataFrame 
df = pd.DataFrame(columns=[
    "sense1_id", "sense1_entry_id", "synset1_id", "sense1_heading", "sense1_gloss", "heading1_PoS",
    "sense2_id", "sense2_entry_id", "synset2_id", "sense2_heading", "sense2_gloss", "heading2_PoS", "rel_type"
])

"""
Because of the database structure we handle synonymy differently than we handle other relations. 
For synonymy we find all unique synset ids and then look for senses that belong to this synset. After that we create a unique relationship between 2 synset members. 
For example, if a synset has n members, we will add a total of 
     (n!)
    ------
  (2!(n-2)!)
rows. 
The other relations (synonymy, hypernymy, also, similar, antonym,meronym) are added based on dict.synset_relations table. If we now that synset b with m members 
is a hypernym of synset a with n members, will add a total of m * n rows. 

Be aware that some of the relationship are symetric and some are not. 
Synonymy, antonymy, similar, also are symetric. 
Meaning that if we would change sense1_id with sense2_id it would not change the relationship type. 
However, it is important that for asymetric relations - hypernym and holonymy - we are consistent with the sequence
of the senses in the dataset. 
For asymetric cases, the hypernym/holonym will always be attained to the columns containing id "2" (synset2_id, sense2_id)
"""

load_dotenv()

# Now you can access these variables using os.getenv
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

try:
    with psycopg2.connect(
        dbname="postgres", user=DB_USER, password=DB_PASSWORD, host="localhost", port="5432"
    ) as conn:
        cursor = conn.cursor()
# Add hypernymy, also, similar, antonym,holonym relationship
        query = """
            SELECT synset_1_id, synset_2_id, synset_rel_types.name
            FROM dict.synset_relations
            JOIN dict.synset_rel_types ON dict.synset_relations.type_id = synset_rel_types.id
        """
        cursor.execute(query)
        rows = cursor.fetchall()
    
        data = []
        for row in tqdm(rows, desc="Processing synset relations(hypernymy, also, similar, antonym,meronym)"):
            synset_1_id, synset_2_id, rel_type = row
            rows_synset_1 = get_synset_members(synset_1_id)
            rows_synset_2 = get_synset_members(synset_2_id)

            # Add other semantic relations
            for row1 in rows_synset_1:
                for row2 in rows_synset_2:
                    data.append(create_row_data(row1, row2, synset_1_id, synset_2_id, rel_type))
# Add synonymy
        query = """
            SELECT DISTINCT synset_id
            FROM dict.senses;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        synset_ids_all = [row[0] for row in rows]
        for synset_id in tqdm(synset_ids_all, desc="Adding synonymy relations"): 
            synset_members  = get_synset_members(synset_id)
            for row1, row2 in itertools.combinations(synset_members, 2):
                data.append(create_row_data(row1, row2, synset_id, synset_id, "synonymy"))

 # Convert collected data to DataFrame
        df = pd.DataFrame(data)

        # Drop duplicates and filter by Part of Speech
        df.drop_duplicates(subset=["sense1_id", "sense2_id"], inplace=True, keep="first")
        df = df[(df['heading1_PoS'] == 'Lietvārds') & (df['heading2_PoS'] == 'Lietvārds')]
        # Save DataFrame to CSV
        df.to_csv("Training process/Dataset/positive_examples_nouns.csv", index=False)
except psycopg2.Error as e:
    print(f"Database connection error: {e}")

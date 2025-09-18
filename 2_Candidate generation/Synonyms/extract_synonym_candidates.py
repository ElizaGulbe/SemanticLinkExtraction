import pandas as pd
import itertools
import psycopg2
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

# Now you can access these variables using os.getenv
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME', 'tezaurs_dv')

# Read the content of the file
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Create a cursor object
cursor = conn.cursor()
def get_syn_relationships():
    with open('Synonyms/syn_dict.txt', 'r') as file:
        content = file.read()

    # Split the content into lines
    lines = content.strip().split("\n")
    synonyms = []

    for line in lines:
        try:
            parts = line.replace(" - ", ",").replace(";", ",").split(",")
            parts = [p.strip() for p in parts]
            combinations = list(itertools.combinations(parts, 2))
            for item in combinations:
                synonyms.append(item)
        except ValueError:
            print(f"Line format issue: {line}")
            continue  # Skip lines that don't have the expected format

    # Convert to DataFrame
    df = pd.DataFrame(synonyms, columns=["heading1", "heading2"])

    # Filter the DataFrame where both headings have exactly one word
    df_filtered = df[(df['heading1'].apply(lambda x: len(x.split())) == 1) & 
                    (df['heading2'].apply(lambda x: len(x.split())) == 1)]

    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered
# Display the first few rows of the filtered DataFrame

dict_syn_candidates = []

def add_row_syn_candidate(row1,row2):

    occurnace_1 = row1[8]["ratioPerMille"] if row1[8] else 0
    occurnace_2 = row2[8]["ratioPerMille"] if row2[8] else 0
    pos_1 = row1[9].get('Vārdšķira') if row1[9] else "None"
    pos_2 = row2[9].get('Vārdšķira') if row2[9] else "None"

    new_row = {
    'sense1_id': row1[0],
    'sense1_entry_id': row1[1],
    'sense1_heading': row1[2],
    'sense1_gloss': row1[4],
    'sense1_pos': pos_1,
    'sense1_parent_sense_id': row1[3],
    'sense1_order_no':row1[5],
    'sense1_tag':row1[6],
    'sense1_hidden': row1[7],
    'sense1_heading_occurance':occurnace_1,
    'sense1_lexemes_extra': row1[10],
    'sense2_id': row2[0],
    'sense2_entry_id': row2[1],
    'sense2_heading': row2[2],
    'sense2_gloss': row2[4],
    'sense2_pos': pos_2,
    'sense2_parent_sense_id': row2[3],
    'sense2_order_no' : row2[5],
    'sense2_tag' : row2[6],
    'sense2_hidden':row2[7],
    'sense2_heading_occurance':occurnace_2,
    'sense2_lexemes_extra': row2[10],
    'sense1_gloss_embedding': None,
    'sense2_gloss_embedding': None
    }
    dict_syn_candidates.append(new_row)

def get_query(heading):
    # Use a single query to retrieve the required data
    query = """
        SELECT 
            senses.id AS sense_id, 
            senses.entry_id, 
            entries.heading AS entry_heading,
            senses.parent_sense_id, 
            senses.gloss, 
            senses.order_no, 
            senses.sense_tag, 
            senses.hidden,
			entries.extras,
            paradigms.data,
			lexemes.data
        FROM 
            dict.senses AS senses
        JOIN 
            dict.entries AS entries
        ON 
            senses.entry_id = entries.id
        LEFT JOIN 
            dict.lexemes AS lexemes
        ON 
            senses.entry_id = lexemes.entry_id
        LEFT JOIN 
            dict.paradigms AS paradigms
        ON 
            lexemes.paradigm_id = paradigms.id
        WHERE 
            entries.heading = %s
    """
    
    cursor.execute(query, (heading,))
    rows = cursor.fetchall()
    return rows

df_source = get_syn_relationships()

for idx,row in tqdm(df_source.iterrows(), total=df_source.shape[0]):
    all_instances_1 = get_query(row["heading1"])
    all_instances_2 = get_query(row["heading2"])
    if not all_instances_1 or not all_instances_2:
        continue
    if len(all_instances_1) > 4 or len(all_instances_2) > 4:
        continue
    for row1 in all_instances_1:
        for row2 in all_instances_2:
            add_row_syn_candidate(row1,row2)



df = pd.DataFrame(dict_syn_candidates)
df = df[(df["sense1_pos"] == "Lietvārds") &(df["sense2_pos"] == "Lietvārds")]
df = df.sort_values(by="sense1_heading_occurance",ascending=False).head(40000)
df = df.sort_values(by="sense1_heading_occurance",ascending=False).head(20000)
df.to_csv("Data selection/synonym_candidates_top10000_4_meaning.csv", index=False)



   







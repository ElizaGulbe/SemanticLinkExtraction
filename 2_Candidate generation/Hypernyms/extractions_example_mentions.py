import pandas as pd
import itertools
import psycopg2
from tqdm import tqdm
import re
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Now you can access these variables using os.getenv
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Read the content of the file
conn = psycopg2.connect(
    dbname="postgres",
    user=DB_USER,
    password=DB_PASSWORD,
    host="localhost",
    port="5432"
)

# Create a cursor object

cursor = conn.cursor()


dict_all = []

def add_row(row1):
    occurnace_1 = row1[8]["ratioPerMille"] if row1[8] else 0
    pos_1 = row1[9].get('Vārdšķira') if row1[9] else "None"
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
    }
    dict_all.append(new_row)

def get_all_queries():
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
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    return rows


hypernym_candidates = []
def add_hypernym_candidate(row1,row2): 
    occurnace_2 = row2["ratioPerMille"] if row2["ratioPerMille"] else 0
    pos_2 = row2['sense1_pos'] if row2['sense1_pos'] else "None"
    new_row = {
    'sense2_id': row1['sense1_id'],
    'sense2_entry_id': row1['sense1_entry_id'],
    'sense2_heading': row1['sense1_heading'],
    'sense2_gloss': row1['sense1_gloss'],
    'sense2_pos': row1['sense1_pos'],
    'sense2_parent_sense_id': row1['sense1_parent_sense_id'],
    'sense2_order_no':row1['sense1_order_no'],
    'sense2_tag':row1['sense1_tag'],
    'sense2_hidden': row1['sense1_hidden'],
    'sense2_heading_occurance':row1['sense1_heading_occurance'],
    'sense2_lexemes_extra': row1['sense1_lexemes_extra'],
    'sense1_id': row2[0],
    'sense1_entry_id': row2[1],
    'sense1_heading': row2[2],
    'sense1_gloss': row2[4],
    'sense1_pos': pos_2,
    'sense1_parent_sense_id': row2[3],
    'sense1_order_no':row2[5],
    'sense1_tag':row2[6],
    'sense1_hidden': row2[7],
    'sense1_heading_occurance':occurnace_2,
    'sense1_lexemes_extra': row2[10]
    }
    hypernym_candidates.append(new_row)

def get_word(heading):
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


# Function to process the entire text and extract the cleaned words
def process_text_virsteikumi(text):
    pattern = r"\((?:piem\.,|piemēram,)\s*([^)]*)\)"
    # Find all matches
    matches = re.findall(pattern, text)
    # Process each match to split by comma and trim whitespace
    hyponyms = []
    for match in matches:
        # Split the matched text by commas and strip extra whitespace
        items = [item.strip() for item in match.split(",")]
        hyponyms.extend(items)  # Add these items to the main list of hyponyms
    return hyponyms
    
def get_number_of_def(df_all): 
    df_all["count_meanings"] = None
    for idx, row in tqdm(df_all.iterrows(), total=df_all.shape[0]):
        entry_id = row["sense1_entry_id"]
        number_of_instances = df_all[df_all["sense1_entry_id"] == entry_id].shape[0]
        df_all.loc[idx,"count_meanings"] = number_of_instances
    return df_all

all_rows = get_all_queries()
for row in tqdm(all_rows):
    add_row(row)

df_all = pd.DataFrame(dict_all)
df_all = get_number_of_def(df_all=df_all)
df_all = df_all[df_all["count_meanings"] < 3]
df_1000_occurance = df_all.nlargest(20000, 'sense1_heading_occurance')
df_1000_occurance = df_1000_occurance[df_1000_occurance["sense1_pos"] == "Lietvārds"]




def check_pos_locijums_criteria(name): 
    if name == "":
        return False
    try:
        url = "http://api.tezaurs.lv:8182/analyze/" + name
        response =requests.get(url)
        if not response.json() or len(response.json()) > 1:
                return False
        for resp in response.json():
            pos = resp["Vārdšķira"]
            if pos != "Lietvārds":
                return False
            locijums = resp["Locījums"]
            if locijums == "Nominatīvs":
                return True
            
        return False
    except requests.exceptions.Timeout:
        print(f"Timeout occurred for {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None



for idx, row in tqdm(df_1000_occurance.iterrows(), total=df_1000_occurance.shape[0]):
    gloss = row["sense1_gloss"]
    candidate_virsteikums = process_text_virsteikumi(gloss)
    for candidate in candidate_virsteikums: 
        if check_pos_locijums_criteria(candidate):

            print(candidate)
            candidate_rows  = get_word(candidate.lower())
            if len(candidate_rows) > 4 :
                continue
            for candidate_row in candidate_rows:
                add_hypernym_candidate(candidate_row,row)
                print("Adding candidate - ") # seit jabut otradak  
                print(row['sense1_heading'] + " is a hypernym for " + candidate_row[2])


df_hypernym = pd.DataFrame(hypernym_candidates)
df_hypernym.to_csv("Data selection/new_laws.csv", index=False)








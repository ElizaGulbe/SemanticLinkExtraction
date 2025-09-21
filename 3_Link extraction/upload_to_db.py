import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import pyarrow # parquet datu formātam

load_dotenv()
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME', 'tezaurs_dv')

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

df = pd.read_parquet('Output/probabilities.parquet')
df = df[['sense1_id', 'sense2_id', 'source', 'hypernym', 'synonymy']]
df = df.rename(columns={'hypernym':'prob_hypernym', 'synonymy':'prob_synonym','source':'candidate_source'})
df['decision'] = pd.Series([pd.NA] * len(df), dtype='string')
df['verified_by'] = pd.Series([pd.NA] * len(df), dtype='string')
df['imported_at'] = pd.Timestamp.now(tz='UTC')

df.info()
df.to_sql(
    "link_candidates",
    engine,
    schema="more",
    if_exists="append", 
    index=False,
    method="multi",
    chunksize=1000
)
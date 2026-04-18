import psycopg2
from sqlalchemy import create_engine, text
import pandas as pd

DB_URL = "postgresql+psycopg2://admin:admin@127.0.0.1:5433/chicago_crimes"
engine = create_engine(DB_URL)

print("Reading Crimes CSV file...")
df = pd.read_csv("Crimes_-_2001_to_Present.csv", low_memory=False)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# Clean column names
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
print("Columns:", df.columns.tolist())

# Create table
print("Creating crimes_raw table...")
df.head(0).to_sql("crimes_raw", engine, if_exists="replace", index=False)

# Load data
print("Wait! Loading data...")
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5433,
    dbname="chicago_crimes",
    user="admin",
    password="admin"
)

import io
buffer = io.StringIO()
df.to_csv(buffer, index=False, header=False)
buffer.seek(0)

cur = conn.cursor()
cur.copy_expert("COPY crimes_raw FROM STDIN WITH CSV", buffer)
conn.commit()
cur.close()
conn.close()

print("Data loaded.")
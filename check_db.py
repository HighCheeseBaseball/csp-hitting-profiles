import sqlite3
import pandas as pd

db_path = "MLB_25.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables found: {tables}")

# Check first table
if tables:
    table_name = tables[0]
    print(f"\nChecking table: {table_name}")
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Columns: {columns[:10]}...")  # First 10 columns
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Row count: {count}")
    
    # Get sample data
    if count > 0:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(f"\nSample data:")
        print(df.head())
        
        # Check for key columns
        key_cols = ['batter_name', 'game_date', 'game_type', 'attack_direction', 'description']
        for col in key_cols:
            if col in df.columns:
                print(f"\n{col} unique values: {df[col].nunique()}")
                if col == 'attack_direction':
                    # Show more details for attack_direction
                    print(f"{col} sample values: {df[col].dropna().unique()[:10]}")
                    print(f"{col} data type: {df[col].dtype}")
                    # Check for a specific batter
                    if 'batter_name' in df.columns:
                        cj_abrams = df[df['batter_name'].str.contains('Abrams', case=False, na=False)]
                        if len(cj_abrams) > 0:
                            inplay_abrams = cj_abrams[cj_abrams['description'] == 'In Play']
                            if len(inplay_abrams) > 0 and 'attack_direction' in inplay_abrams.columns:
                                print(f"\nCJ Abrams attack_direction (in play):")
                                print(f"  Sample values: {inplay_abrams['attack_direction'].dropna().head(10).tolist()}")
                                print(f"  Mean: {inplay_abrams['attack_direction'].dropna().mean()}")
                                print(f"  Count: {inplay_abrams['attack_direction'].notna().sum()}")
                else:
                    print(f"{col} sample: {df[col].unique()[:5]}")

conn.close()


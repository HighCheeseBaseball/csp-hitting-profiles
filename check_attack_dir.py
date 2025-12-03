import pandas as pd
import os
from data_utils import preprocess_data

# Find CSV file
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if csv_files:
    csv_path = csv_files[0]
    print(f"Loading: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False, nrows=100000)  # Load first 100k rows for testing
    # Apply preprocessing to match what the app does
    df = preprocess_data(df)
    
    # Check for CJ Abrams
    if 'batter_name' in df.columns:
        abrams = df[df['batter_name'].str.contains('Abrams', case=False, na=False)]
        print(f"\nCJ Abrams rows found: {len(abrams)}")
        
        if len(abrams) > 0:
            # Check attack_direction
            if 'attack_direction' in abrams.columns:
                print(f"\nAttack Direction column info:")
                print(f"  Data type: {abrams['attack_direction'].dtype}")
                print(f"  Non-null count: {abrams['attack_direction'].notna().sum()}")
                print(f"  Sample values: {abrams['attack_direction'].dropna().head(20).tolist()}")
                
                # Check description values
                if 'description' in abrams.columns:
                    print(f"\nDescription values for Abrams:")
                    print(f"  Unique: {abrams['description'].unique()}")
                    print(f"  Value counts:\n{abrams['description'].value_counts()}")
                    
                    # Check for any rows with attack_direction
                    has_attack_dir = abrams[abrams['attack_direction'].notna()]
                    print(f"\nRows with attack_direction: {len(has_attack_dir)}")
                    if len(has_attack_dir) > 0:
                        print(f"  Their descriptions: {has_attack_dir['description'].value_counts()}")
                        print(f"  Mean attack_direction (all): {has_attack_dir['attack_direction'].mean()}")
                        
                        # Check if it's based on events instead
                        if 'events' in abrams.columns:
                            print(f"\nEvents for rows with attack_direction:")
                            print(has_attack_dir['events'].value_counts())
                            
                            # Maybe it's based on events that are hits?
                            # Check both original and recoded event names
                            hits1 = has_attack_dir[has_attack_dir['events'].isin(['Single', 'Double', 'Triple', 'Home Run'])]
                            hits2 = has_attack_dir[has_attack_dir['events'].isin(['single', 'double', 'triple', 'home_run'])]
                            hits = pd.concat([hits1, hits2]).drop_duplicates()
                            print(f"\nHits with attack_direction: {len(hits)}")
                            if len(hits) > 0:
                                print(f"  Mean: {hits['attack_direction'].mean()}")
                                print(f"  Events: {hits['events'].value_counts()}")
                            
                            # Check all "In Play" events (including outs)
                            if 'description' in has_attack_dir.columns:
                                inplay_all = has_attack_dir[has_attack_dir['description'] == 'In Play']
                                print(f"\nAll 'In Play' with attack_direction: {len(inplay_all)}")
                                if len(inplay_all) > 0:
                                    print(f"  Mean: {inplay_all['attack_direction'].mean()}")
                                    print(f"  Events breakdown: {inplay_all['events'].value_counts()}")
            else:
                print("attack_direction column not found")
                print(f"Available columns: {[c for c in abrams.columns if 'attack' in c.lower() or 'direction' in c.lower()]}")
    else:
        print("batter_name column not found")


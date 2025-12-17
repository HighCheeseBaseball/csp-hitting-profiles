import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from data_utils import preprocess_data
from io import BytesIO
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Hitter Profiles",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - White background with black text
st.markdown("""
    <style>
    /* White background for main content */
    .stApp {
        background-color: #ffffff;
    }
    .main .block-container {
        background-color: #ffffff;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-weight: 600;
        color: #495057;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #212529;
    }
    /* Compact tables with black borders */
    .stDataFrame {
        font-size: 0.85rem;
    }
    .stDataFrame table {
        margin-bottom: 0.5rem !important;
        border: 1px solid #000000 !important;
    }
    .stDataFrame thead th {
        padding: 0.3rem 0.5rem !important;
        font-size: 0.85rem !important;
        border: 1px solid #000000 !important;
        color: #000000 !important;
    }
    .stDataFrame tbody td {
        padding: 0.3rem 0.5rem !important;
        font-size: 0.85rem !important;
        border: 1px solid #000000 !important;
        color: #000000 !important;
    }
    .stDataFrame tbody tr {
        border-bottom: 2px solid #000000 !important;
    }
    /* Compact section headers - black text */
    h4 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.1rem !important;
        font-size: 0.95rem !important;
        text-align: center !important;
        color: #000000 !important;
    }
    h3 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
        color: #000000 !important;
    }
    h2 {
        color: #000000 !important;
    }
    /* All text black */
    p, div, span, label {
        color: #000000 !important;
    }
    /* Reduce spacing between sections */
    hr {
        margin-top: 0.1rem !important;
        margin-bottom: 0.3rem !important;
        border-color: #000000 !important;
    }
    /* Reduce table spacing */
    .stDataFrame {
        margin-bottom: 0.2rem !important;
    }
    .stDataFrame table {
        margin-bottom: 0.2rem !important;
    }
    /* Sidebar styling - white text */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] input {
        color: #ffffff !important;
    }
    /* Notes section styling */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stTextArea label {
        color: #000000 !important;
    }
    /* Save button styling - white background */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }
    .stButton > button:hover {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'profile_notes' not in st.session_state:
    st.session_state.profile_notes = {}

# Data loading functions
@st.cache_data
def calculate_league_hc_x_thresholds(df):
    """Calculate league-wide hc_x thresholds for Pull/Center/Oppo zones"""
    if df is None or len(df) == 0:
        return {'center_min': 120, 'center_max': 154}  # Fallback defaults
    
    inplay = df[df['description'] == 'In Play'] if 'description' in df.columns else pd.DataFrame()
    if len(inplay) == 0 or 'hc_x' not in inplay.columns:
        return {'center_min': 120, 'center_max': 154}  # Fallback defaults
    
    hc_x = inplay['hc_x'].dropna()
    if len(hc_x) == 0:
        return {'center_min': 120, 'center_max': 154}  # Fallback defaults
    
    # Use thirds of league distribution
    lower_bound = hc_x.quantile(0.333)
    upper_bound = hc_x.quantile(0.667)
    
    return {
        'center_min': lower_bound,
        'center_max': upper_bound
    }

@st.cache_data
def load_data(csv_path=None):
    """Load data from CSV file - prioritize MLB_data.csv"""
    import urllib.request
    
    # Get the current working directory
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find CSV file
    if csv_path is None:
        # Look for MLB_data.csv first (main data file)
        possible_names = [
            "MLB_data.csv",  # Prioritize this
            "MLB25.csv",
            "MLB_25.csv", 
            "data.csv",
            "hitter_data.csv",
            "pitch_data.csv"
        ]
        
        csv_path = None
        for name in possible_names:
            # Try current directory first
            full_path = os.path.join(current_dir, name)
            if os.path.exists(full_path):
                csv_path = full_path
                break
            # Try script directory
            full_path = os.path.join(script_dir, name)
            if os.path.exists(full_path):
                csv_path = full_path
                break
        
        # If MLB_data.csv not found, try downloading from cloud storage
        if csv_path is None:
            # Check for download URL in environment variable or secrets
            try:
                download_url = os.getenv('MLB_DATA_URL')
                if not download_url:
                    try:
                        download_url = st.secrets.get('mlb_data_url', None)
                    except:
                        download_url = None
                
                if download_url:
                    csv_path = os.path.join(current_dir, "MLB_data.csv")
                    if not os.path.exists(csv_path) or (os.path.exists(csv_path) and os.path.getsize(csv_path) < 1000):
                        with st.spinner("Downloading data file from Google Drive (this may take a few minutes)..."):
                            try:
                                import gdown
                                import re
                                
                                # Extract file ID from Google Drive URL
                                file_id = None
                                if 'drive.google.com' in download_url:
                                    # Extract ID from various Google Drive URL formats
                                    match = re.search(r'/d/([a-zA-Z0-9_-]+)', download_url)
                                    if match:
                                        file_id = match.group(1)
                                    
                                    if file_id:
                                        # Use gdown to download (handles large files and virus scan warnings)
                                        gdown_url = f"https://drive.google.com/uc?id={file_id}"
                                        gdown.download(gdown_url, csv_path, quiet=False)
                                        
                                        # Verify download
                                        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
                                            file_size_mb = os.path.getsize(csv_path) / (1024*1024)
                                            st.success(f"Data file downloaded successfully! ({file_size_mb:.1f} MB)")
                                        else:
                                            st.error("Download failed - file is too small or empty.")
                                            if os.path.exists(csv_path):
                                                os.remove(csv_path)
                                            return None
                                    else:
                                        st.error("Could not extract file ID from Google Drive URL.")
                                        return None
                                else:
                                    # Not a Google Drive URL, use regular download
                                    import requests
                                    response = requests.get(download_url, stream=True)
                                    with open(csv_path, 'wb') as f:
                                        for chunk in response.iter_content(chunk_size=8192):
                                            if chunk:
                                                f.write(chunk)
                                    if os.path.getsize(csv_path) > 1000:
                                        file_size_mb = os.path.getsize(csv_path) / (1024*1024)
                                        st.success(f"Data file downloaded successfully! ({file_size_mb:.1f} MB)")
                                    else:
                                        st.error("Download failed - file is too small.")
                                        return None
                            except Exception as e:
                                st.error(f"Download failed: {str(e)}")
                                if os.path.exists(csv_path):
                                    os.remove(csv_path)
                                return None
                    else:
                        file_size = os.path.getsize(csv_path) / (1024*1024)
                        st.info(f"Using cached data file ({file_size:.1f} MB).")
            except Exception as e:
                st.error(f"Could not download data file: {str(e)}")
                st.info("Please check that MLB_DATA_URL is set correctly in Railway variables.")
                import traceback
                st.code(traceback.format_exc())
        
        if csv_path is None:
            # List CSV files in directory
            try:
                csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv') and f != 'Bat_path.csv']
                if csv_files:
                    csv_path = os.path.join(current_dir, csv_files[0])  # Use first CSV found (excluding Bat_path.csv)
                else:
                    # Try the script directory
                    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv') and f != 'Bat_path.csv']
                    if csv_files:
                        csv_path = os.path.join(script_dir, csv_files[0])
                    else:
                        st.error(f"No CSV file found in current directory: {current_dir}")
                        st.info(f"Please ensure MLB_data.csv (or another CSV file) is in the same directory as the app.")
                        st.info(f"Looking for files: {', '.join(possible_names)}")
                        return None
            except Exception as e:
                st.error(f"Error searching for CSV files: {e}")
                return None
    
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        st.info(f"Current directory: {current_dir}")
        return None
    
    try:
        # Try different encodings and separators
        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, sep=sep, low_memory=False)
                    if len(df) > 0:
                                        # Don't show success message
                        break
                except:
                    continue
            if df is not None and len(df) > 0:
                break
        
        if df is None or len(df) == 0:
            st.error("Could not read CSV file or file is empty")
            return None
        
        # Convert game_date to datetime - handle different date formats and column names
        date_col = None
        possible_date_cols = ['game_date', 'Game_Date', 'GAME_DATE', 'date', 'Date', 'DATE', 'gameDate', 'GameDate', 'game_date_utc']
        
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                # Try parsing as datetime first
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # If that fails, try Excel date serial number
                if df[date_col].isna().all():
                    # Check if original column is numeric (Excel serial date)
                    original_series = df[date_col]
                    if pd.api.types.is_numeric_dtype(original_series):
                        df[date_col] = pd.to_datetime('1899-12-30') + pd.to_timedelta(original_series, unit='D')
                
                # Filter out invalid dates
                df = df[df[date_col].notna()]
                # Filter out dates before 2000 (likely errors)
                df = df[df[date_col] >= pd.Timestamp('2000-01-01')]
                
                # Rename to 'game_date' for consistency
                if date_col != 'game_date':
                    df = df.rename(columns={date_col: 'game_date'})
            except Exception as e:
                pass  # Silently handle date parsing errors
        
        # Filter for post-season game types if column exists
        if 'game_type' in df.columns:
            game_types_post = ['R', 'F', 'D', 'L', 'W']
            df = df[df['game_type'].isin(game_types_post)]
        
        if len(df) == 0:
            st.error("No data remaining after filtering")
            return None
        
        # Preprocess data to add derived columns
        df = preprocess_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

@st.cache_data
def calculate_percentile_references(df):
    """Calculate reference percentiles for all batters, split by overall, RHP, and LHP"""
    if len(df) == 0:
        return {'overall': {}, 'rhp': {}, 'lhp': {}}
    
    # Calculate league-wide thresholds once
    thresholds = calculate_league_hc_x_thresholds(df)
    
    # Split by pitcher handedness
    df_rhp = df[df['p_throws'] == 'R'] if 'p_throws' in df.columns else pd.DataFrame()
    df_lhp = df[df['p_throws'] == 'L'] if 'p_throws' in df.columns else pd.DataFrame()
    
    def calc_references_for_subset(subset_df, subset_name, thresholds):
        """Calculate reference percentiles for a subset of data"""
        references = {}
        
        if len(subset_df) == 0:
            return references
        
        # Group by batter to get per-batter stats
        batter_groups = subset_df.groupby('batter_name')
        
        # Calculate stats for each batter
        batter_stats_list = []
        for batter_name, group_df in batter_groups:
            stats = {}
            stats['batter_name'] = batter_name
            
            # Basic counting stats
            stats['PA'] = group_df['is_PA'].sum() if 'is_PA' in group_df.columns else 0
            stats['AB'] = group_df['is_ab'].sum() if 'is_ab' in group_df.columns else 0
            stats['H'] = group_df['is_hit'].sum() if 'is_hit' in group_df.columns else 0
            
            # Count hits by type
            singles = (group_df['events'] == 'Single').sum() if 'events' in group_df.columns else 0
            doubles = (group_df['events'] == 'Double').sum() if 'events' in group_df.columns else 0
            triples = (group_df['events'] == 'Triple').sum() if 'events' in group_df.columns else 0
            stats['HR'] = (group_df['events'] == 'Home Run').sum() if 'events' in group_df.columns else 0
            stats['XBH'] = doubles + triples + stats['HR']
            
            # Calculate AVG
            stats['AVG'] = stats['H'] / stats['AB'] if stats['AB'] > 0 else None
            
            # Calculate OBP
            BB = (group_df['events'] == 'Walk').sum() if 'events' in group_df.columns else 0
            HBP = (group_df['events'] == 'HBP').sum() if 'events' in group_df.columns else 0
            SF = group_df['events'].isin(['Sac Fly', 'Sac Fly DP']).sum() if 'events' in group_df.columns else 0
            denom = stats['AB'] + BB + HBP + SF
            stats['OBP'] = (stats['H'] + BB + HBP) / denom if denom > 0 else None
            
            # Calculate SLG correctly: Total Bases / AB
            # Total Bases = (Singles × 1) + (Doubles × 2) + (Triples × 3) + (Home Runs × 4)
            total_bases = singles + (doubles * 2) + (triples * 3) + (stats['HR'] * 4)
            stats['SLG'] = total_bases / stats['AB'] if stats['AB'] > 0 else None
            stats['OPS'] = (stats['OBP'] + stats['SLG']) if (stats['OBP'] is not None and stats['SLG'] is not None) else None
            
            # wOBA metrics
            stats['wOBA'] = group_df['woba_value'].mean() if 'woba_value' in group_df.columns else None
            stats['xwOBA'] = group_df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in group_df.columns else None
            
            # K% and BB%
            k_pct = None
            bb_pct = None
            if stats['PA'] > 0:
                strikeouts = (group_df['events'] == 'Strikeout').sum() if 'events' in group_df.columns else 0
                k_pct = (strikeouts / stats['PA']) * 100
                bb_pct = (BB / stats['PA']) * 100
            stats['KPct'] = k_pct
            stats['BBPct'] = bb_pct
            
            # Launch metrics (only for in-play)
            inplay = group_df[group_df['description'] == 'In Play'] if 'description' in group_df.columns else pd.DataFrame()
            stats['LaunchSpeed'] = inplay['launch_speed'].mean() if len(inplay) > 0 and 'launch_speed' in inplay.columns else None
            stats['LaunchAngle'] = inplay['launch_angle'].mean() if len(inplay) > 0 and 'launch_angle' in inplay.columns else None
            stats['ExitVelo'] = stats['LaunchSpeed']  # Same as LaunchSpeed
            
            # 90th percentile EV
            if len(inplay) > 0 and 'launch_speed' in inplay.columns:
                sorted_ev = inplay['launch_speed'].dropna().sort_values(ascending=False)
                if len(sorted_ev) > 0:
                    top_10_pct = int(max(1, len(sorted_ev) * 0.1))
                    stats['EV_90th'] = sorted_ev.head(top_10_pct).mean()
                else:
                    stats['EV_90th'] = None
            else:
                stats['EV_90th'] = None
            
            # Max Exit Velo
            if len(inplay) > 0 and 'launch_speed' in inplay.columns:
                stats['MaxEV'] = inplay['launch_speed'].max()
            else:
                stats['MaxEV'] = None
            
            # Launch Angle Sweet Spot %
            if len(inplay) > 0 and 'launch_angle' in inplay.columns:
                sweet_spot = ((inplay['launch_angle'] >= 8) & (inplay['launch_angle'] <= 32)).sum()
                stats['LASweetSpotPct'] = (sweet_spot / len(inplay)) * 100 if len(inplay) > 0 else None
            else:
                stats['LASweetSpotPct'] = None
            
            # ISO
            stats['ISO'] = inplay['iso_value'].mean() if len(inplay) > 0 and 'iso_value' in inplay.columns else None
            
            # Hard Hit %
            if len(inplay) > 0 and 'launch_speed' in inplay.columns:
                hard_hit = (inplay['launch_speed'] >= 95).sum()
                stats['HardHitPct'] = (hard_hit / len(inplay)) * 100
            else:
                stats['HardHitPct'] = None
            
            # Barrel %
            if len(inplay) > 0 and 'BBECheck' in inplay.columns and 'is_barrel' in inplay.columns:
                bbe = inplay['BBECheck'].sum()
                barrels = inplay['is_barrel'].sum()
                stats['BarrelPct'] = (barrels / bbe * 100) if bbe > 0 else None
            else:
                stats['BarrelPct'] = None
            
            # Batted ball type percentages
            if len(inplay) > 0 and 'bb_type' in inplay.columns:
                total_bbe = len(inplay)
                stats['GBPct'] = ((inplay['bb_type'] == 'ground_ball').sum() / total_bbe) * 100 if total_bbe > 0 else None
                stats['LDPct'] = ((inplay['bb_type'] == 'line_drive').sum() / total_bbe) * 100 if total_bbe > 0 else None
                stats['FBPct'] = ((inplay['bb_type'] == 'fly_ball').sum() / total_bbe) * 100 if total_bbe > 0 else None
            else:
                stats['GBPct'] = None
                stats['LDPct'] = None
                stats['FBPct'] = None
            
            # Pull/Center/Oppo percentages (need to calculate using hc_x and stand)
            try:
                if len(inplay) > 0 and 'hc_x' in inplay.columns and 'stand' in inplay.columns:
                    # Use league-wide thresholds passed as parameter
                    center_min = thresholds.get('center_min', 120)
                    center_max = thresholds.get('center_max', 154)
                    
                    valid_data = inplay[['hc_x', 'stand']].dropna()
                    if len(valid_data) > 0:
                        total_bbe = len(valid_data)
                        pull_count = 0
                        center_count = 0
                        oppo_count = 0
                        
                        for idx, row in valid_data.iterrows():
                            try:
                                hc_x_val = row['hc_x']
                                batter_stand = row['stand']
                                
                                if center_min <= hc_x_val <= center_max:
                                    center_count += 1
                                else:
                                    if batter_stand == 'R':
                                        if hc_x_val < center_min:
                                            pull_count += 1
                                        else:
                                            oppo_count += 1
                                    else:  # L
                                        if hc_x_val > center_max:
                                            pull_count += 1
                                        else:
                                            oppo_count += 1
                            except:
                                continue
                        
                        stats['PullPct'] = (pull_count / total_bbe) * 100 if total_bbe > 0 else None
                        stats['CenterPct'] = (center_count / total_bbe) * 100 if total_bbe > 0 else None
                        stats['OppoPct'] = (oppo_count / total_bbe) * 100 if total_bbe > 0 else None
                    else:
                        stats['PullPct'] = None
                        stats['CenterPct'] = None
                        stats['OppoPct'] = None
                else:
                    stats['PullPct'] = None
                    stats['CenterPct'] = None
                    stats['OppoPct'] = None
                
                # Pull GB% and Pull AIR%
                if len(inplay) > 0 and 'hc_x' in inplay.columns and 'stand' in inplay.columns and 'bb_type' in inplay.columns:
                    center_min = thresholds.get('center_min', 120)
                    center_max = thresholds.get('center_max', 154)
                    
                    valid_data = inplay[['hc_x', 'stand', 'bb_type']].dropna()
                    if len(valid_data) > 0:
                        total_bbe = len(valid_data)
                        pulled_gb_count = 0
                        pulled_air_count = 0
                        
                        for idx, row in valid_data.iterrows():
                            try:
                                hc_x_val = row['hc_x']
                                batter_stand = row['stand']
                                bb_type = row['bb_type']
                                
                                is_pulled = False
                                if batter_stand == 'R':
                                    if hc_x_val < center_min:
                                        is_pulled = True
                                else:  # L
                                    if hc_x_val > center_max:
                                        is_pulled = True
                                
                                if is_pulled:
                                    if bb_type == 'ground_ball':
                                        pulled_gb_count += 1
                                    elif bb_type in ['fly_ball', 'line_drive']:
                                        pulled_air_count += 1
                            except:
                                continue
                        
                        stats['PullGBPct'] = (pulled_gb_count / total_bbe) * 100 if total_bbe > 0 else None
                        stats['PullAIRPct'] = (pulled_air_count / total_bbe) * 100 if total_bbe > 0 else None
                    else:
                        stats['PullGBPct'] = None
                        stats['PullAIRPct'] = None
                else:
                    stats['PullGBPct'] = None
                    stats['PullAIRPct'] = None
            except Exception as e:
                # If there's any error, set to None
                stats['PullPct'] = None
                stats['CenterPct'] = None
                stats['OppoPct'] = None
                stats['PullGBPct'] = None
                stats['PullAIRPct'] = None
            
            # Bat path metrics
            stats['AttackAngle'] = group_df['attack_angle'].mean() if 'attack_angle' in group_df.columns else None
            stats['SwingTilt'] = group_df['swing_path_tilt'].mean() if 'swing_path_tilt' in group_df.columns else None
            
            # Swing Length (from main dataframe)
            stats['SwingLength'] = group_df['swing_length'].mean() if 'swing_length' in group_df.columns else None
            
            # Contact metrics
            total = len(group_df)
            stats['SwingPct'] = (group_df['is_swing'].sum() / total * 100) if total > 0 and 'is_swing' in group_df.columns else None
            stats['ZonePct'] = (group_df['is_in_zone'].mean() * 100) if 'is_in_zone' in group_df.columns else None
            
            out_zone = group_df['is_out_zone'].sum() if 'is_out_zone' in group_df.columns else 0
            stats['ChasePct'] = (group_df['is_chase'].sum() / out_zone * 100) if out_zone > 0 else None
            
            swings = group_df['is_swing'].sum() if 'is_swing' in group_df.columns else 0
            stats['WhiffPct'] = (group_df['is_whiff'].sum() / swings * 100) if swings > 0 else None
            
            # Zone Whiff %
            zone_swings = group_df['is_zone_swing'].sum() if 'is_zone_swing' in group_df.columns else 0
            stats['ZoneWhiffPct'] = (group_df['is_zone_whiff'].sum() / zone_swings * 100) if zone_swings > 0 else None
            
            batter_stats_list.append(stats)
        
        batter_stats = pd.DataFrame(batter_stats_list)
        
        # Store reference values (drop NaN values)
        for key in ['PA', 'AB', 'H', 'HR', 'XBH', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'xwOBA',
                    'KPct', 'BBPct',
                    'LaunchSpeed', 'LaunchAngle', 'ExitVelo', 'EV_90th', 'MaxEV', 'ISO', 'HardHitPct', 'BarrelPct',
                    'GBPct', 'LDPct', 'FBPct', 'PullPct', 'CenterPct', 'OppoPct', 'PullGBPct', 'PullAIRPct',
                    'LASweetSpotPct', 'AttackAngle', 'SwingTilt', 'SwingLength', 'SwingPct', 'ZonePct', 'ChasePct', 'WhiffPct', 'ZoneWhiffPct']:
            if key in batter_stats.columns:
                ref_values = batter_stats[key].dropna().values
                if len(ref_values) > 0:
                    references[key] = ref_values
        
        return references
    
    return {
        'overall': calc_references_for_subset(df, 'overall', thresholds),
        'rhp': calc_references_for_subset(df_rhp, 'rhp', thresholds),
        'lhp': calc_references_for_subset(df_lhp, 'lhp', thresholds)
    }

def format_metric(value, decimals=2):
    """Format metric value"""
    if pd.isna(value) or value is None:
        return "-"
    return f"{value:.{decimals}f}"

def format_percent(value, decimals=1):
    """Format percentage value"""
    if pd.isna(value) or value is None:
        return "-"
    return f"{value:.{decimals}f}%"

def format_batter_name(name):
    """Convert 'Last, First' to 'First Last'"""
    if pd.isna(name) or name is None:
        return name
    name_str = str(name).strip()
    if ',' in name_str:
        parts = [p.strip() for p in name_str.split(',', 1)]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    return name_str

def calculate_percentile(value, reference_values, reverse=False):
    """Calculate percentile for a value given reference values
    
    Args:
        value: The value to calculate percentile for
        reference_values: List/array of reference values
        reverse: If True, lower values = higher percentile (for metrics where lower is better)
                 If False, higher values = higher percentile (default, for metrics where higher is better)
    
    Returns:
        Percentile (0-100) representing what percentage of reference values are <= (or >= if reverse) the given value
    """
    if pd.isna(value) or value is None:
        return None
    
    if reference_values is None:
        return None
    
    # Convert to list and filter out NaN values
    if isinstance(reference_values, (pd.Series, np.ndarray)):
        reference_values = reference_values.tolist()
    elif not isinstance(reference_values, list):
        try:
            reference_values = list(reference_values)
        except:
            return None
    
    # Filter out NaN and None values
    reference_values = [v for v in reference_values if not (pd.isna(v) or v is None)]
    
    if len(reference_values) == 0:
        return None
    
    # Count how many reference values are <= (or >= if reverse) the given value
    if reverse:
        # For reverse metrics (lower is better): count values >= given value
        count = sum(1 for v in reference_values if v >= value)
    else:
        # For normal metrics (higher is better): count values <= given value
        count = sum(1 for v in reference_values if v <= value)
    
    percentile = (count / len(reference_values)) * 100
    
    # Ensure percentile is between 0 and 100
    percentile = max(0, min(100, percentile))
    
    return percentile

def get_color_for_percentile(percentile):
    """Get color based on percentile using smooth gradient (dark blue = bad, white = average, dark red = good)"""
    if percentile is None:
        return "#ffffff"  # White for no data
    
    # Clamp percentile to 0-100
    percentile = max(0, min(100, percentile))
    
    # Create smooth gradient from dark blue (0%) -> white (50%) -> dark red (100%)
    # Using smoother interpolation with easing
    # Dark blue: #1e3a8a (30, 58, 138)
    # White: #ffffff (255, 255, 255)
    # Dark red: #991b1b (153, 27, 27)
    
    if percentile <= 50:
        # Interpolate from dark blue to white (0% to 50%)
        # Use smooth easing function for cleaner transition
        ratio = percentile / 50.0
        # Apply easing for smoother transition
        eased_ratio = ratio * ratio * (3.0 - 2.0 * ratio)  # Smoothstep function
        r = int(30 + (255 - 30) * eased_ratio)
        g = int(58 + (255 - 58) * eased_ratio)
        b = int(138 + (255 - 138) * eased_ratio)
    else:
        # Interpolate from white to dark red (50% to 100%)
        ratio = (percentile - 50) / 50.0
        # Apply easing for smoother transition
        eased_ratio = ratio * ratio * (3.0 - 2.0 * ratio)  # Smoothstep function
        r = int(255 - (255 - 153) * eased_ratio)
        g = int(255 - (255 - 27) * eased_ratio)
        b = int(255 - (255 - 27) * eased_ratio)
    
    # Ensure values are in valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return f"#{r:02x}{g:02x}{b:02x}"

def calculate_offensive_metrics(df, references=None):
    """Calculate offensive statistics with LHP/RHP splits and percentiles"""
    if len(df) == 0:
        return pd.DataFrame({
            "Metric": ["No data"], 
            "Overall": ["-"], 
            "vs RHP": ["-"], 
            "vs LHP": ["-"]
        })
    
    # Split by pitcher handedness
    df_rhp = df[df['p_throws'] == 'R'] if 'p_throws' in df.columns else pd.DataFrame()
    df_lhp = df[df['p_throws'] == 'L'] if 'p_throws' in df.columns else pd.DataFrame()
    
    def calc_offensive_for_subset(subset_df):
        """Calculate offensive metrics for a subset"""
        if len(subset_df) == 0:
            return {}
        
        PA = subset_df['is_PA'].sum() if 'is_PA' in subset_df.columns else 0
        AB = subset_df['is_ab'].sum() if 'is_ab' in subset_df.columns else 0
        H = subset_df['is_hit'].sum() if 'is_hit' in subset_df.columns else 0
        
        # Count hits by type
        singles = (subset_df['events'] == 'Single').sum() if 'events' in subset_df.columns else 0
        doubles = (subset_df['events'] == 'Double').sum() if 'events' in subset_df.columns else 0
        triples = (subset_df['events'] == 'Triple').sum() if 'events' in subset_df.columns else 0
        HR = (subset_df['events'] == 'Home Run').sum() if 'events' in subset_df.columns else 0
        XBH = doubles + triples + HR
        
        BB = (subset_df['events'] == 'Walk').sum() if 'events' in subset_df.columns else 0
        HBP = (subset_df['events'] == 'HBP').sum() if 'events' in subset_df.columns else 0
        SF = subset_df['events'].isin(['Sac Fly', 'Sac Fly DP']).sum() if 'events' in subset_df.columns else 0
        
        avg = H / AB if AB > 0 else None
        obp = (H + BB + HBP) / (AB + BB + HBP + SF) if (AB + BB + HBP + SF) > 0 else None
        
        # Calculate SLG correctly: Total Bases / AB
        # Total Bases = (Singles × 1) + (Doubles × 2) + (Triples × 3) + (Home Runs × 4)
        total_bases = singles + (doubles * 2) + (triples * 3) + (HR * 4)
        slg = total_bases / AB if AB > 0 else None
        
        ops = (obp + slg) if (obp is not None and slg is not None) else None
        woba = subset_df['woba_value'].mean() if 'woba_value' in subset_df.columns else None
        xwoba = subset_df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in subset_df.columns else None
        
        # K% and BB%
        k_pct = None
        bb_pct = None
        if PA > 0:
            strikeouts = (subset_df['events'] == 'Strikeout').sum() if 'events' in subset_df.columns else 0
            k_pct = (strikeouts / PA) * 100
            bb_pct = (BB / PA) * 100
        
        return {
            'PA': PA, 'AB': AB, 'H': H, 'HR': HR, 'XBH': XBH,
            'AVG': avg, 'OBP': obp, 'SLG': slg, 'OPS': ops, 'wOBA': woba, 'xwOBA': xwoba,
            'K%': k_pct, 'BB%': bb_pct
        }
    
    overall_metrics = calc_offensive_for_subset(df)
    rhp_metrics = calc_offensive_for_subset(df_rhp)
    lhp_metrics = calc_offensive_for_subset(df_lhp)
    
    result = []
    # Define which metrics should have reversed percentiles (lower is better)
    reversed_metrics = {"KPct"}  # K% - lower is better
    
    metric_definitions = [
        ("PA", "PA", None, "PA", False),
        ("AB", "AB", None, "AB", False),
        ("H", "H", None, "H", False),
        ("HR", "HR", None, "HR", False),
        ("XBH", "XBH", None, "XBH", False),
        ("AVG", "AVG", 3, "AVG", False),
        ("OBP", "OBP", 3, "OBP", False),
        ("SLG", "SLG", 3, "SLG", False),
        ("OPS", "OPS", 3, "OPS", False),
        ("wOBA", "wOBA", 3, "wOBA", False),
        ("xwOBA", "xwOBA", 3, "xwOBA", False),
        ("K%", "K%", 1, "KPct", True),  # Lower is better
        ("BB%", "BB%", 1, "BBPct", False),
    ]
    
    for name, key, decimals, ref_key, reverse in metric_definitions:
        overall_val = overall_metrics.get(key)
        rhp_val = rhp_metrics.get(key)
        lhp_val = lhp_metrics.get(key)
        
        # Calculate percentiles
        overall_pct = None
        rhp_pct = None
        lhp_pct = None
        
        if references and ref_key:
            if 'overall' in references and ref_key in references['overall'] and overall_val is not None:
                overall_pct = calculate_percentile(overall_val, references['overall'][ref_key], reverse=reverse)
            if 'rhp' in references and ref_key in references['rhp'] and rhp_val is not None:
                rhp_pct = calculate_percentile(rhp_val, references['rhp'][ref_key], reverse=reverse)
            if 'lhp' in references and ref_key in references['lhp'] and lhp_val is not None:
                lhp_pct = calculate_percentile(lhp_val, references['lhp'][ref_key], reverse=reverse)
        
        if name in ["PA", "AB", "H", "HR", "XBH"]:
            overall_formatted = str(int(overall_val)) if overall_val is not None else "-"
            rhp_formatted = str(int(rhp_val)) if rhp_val is not None else "-"
            lhp_formatted = str(int(lhp_val)) if lhp_val is not None else "-"
        elif name.endswith("%"):
            # Format as percentage
            overall_formatted = format_percent(overall_val, 1) if overall_val is not None else "-"
            rhp_formatted = format_percent(rhp_val, 1) if rhp_val is not None else "-"
            lhp_formatted = format_percent(lhp_val, 1) if lhp_val is not None else "-"
        else:
            overall_formatted = format_metric(overall_val, decimals) if overall_val is not None else "-"
            rhp_formatted = format_metric(rhp_val, decimals) if rhp_val is not None else "-"
            lhp_formatted = format_metric(lhp_val, decimals) if lhp_val is not None else "-"
        
        result.append({
            "Metric": name,
            "Overall": overall_formatted,
            "Overall_Pct": overall_pct,
            "vs RHP": rhp_formatted,
            "vs RHP_Pct": rhp_pct,
            "vs LHP": lhp_formatted,
            "vs LHP_Pct": lhp_pct
        })
    
    return pd.DataFrame(result)

def calculate_ball_in_play_metrics(df, references=None, full_df=None):
    """Calculate combined ball flight and quality of contact metrics with LHP/RHP splits"""
    inplay = df[df['description'] == 'In Play'] if 'description' in df.columns else pd.DataFrame()
    
    if len(inplay) == 0:
        return pd.DataFrame({
            "Metric": ["No in-play events"], 
            "Overall": ["-"], 
            "vs RHP": ["-"], 
            "vs LHP": ["-"]
        })
    
    # Calculate league-wide thresholds for Pull/Center/Oppo
    if full_df is not None:
        thresholds = calculate_league_hc_x_thresholds(full_df)
    else:
        thresholds = {'center_min': 120, 'center_max': 154}  # Fallback defaults
    
    # Split by pitcher handedness
    inplay_rhp = inplay[inplay['p_throws'] == 'R'] if 'p_throws' in inplay.columns else pd.DataFrame()
    inplay_lhp = inplay[inplay['p_throws'] == 'L'] if 'p_throws' in inplay.columns else pd.DataFrame()
    
    # Split full df by pitcher handedness for ISO calculation (needs all ABs, not just in-play)
    df_rhp = df[df['p_throws'] == 'R'] if 'p_throws' in df.columns else pd.DataFrame()
    df_lhp = df[df['p_throws'] == 'L'] if 'p_throws' in df.columns else pd.DataFrame()
    
    def calc_metrics_for_subset(subset_df, is_percentile=False):
        """Calculate metrics for a subset of data"""
        if len(subset_df) == 0:
            return {}
        
        metrics_dict = {}
        center_min = thresholds['center_min']
        center_max = thresholds['center_max']
        
        # Exit velocity metrics
        if 'launch_speed' in subset_df.columns:
            metrics_dict['avg_ev'] = subset_df['launch_speed'].mean()
            metrics_dict['max_ev'] = subset_df['launch_speed'].max()
            
            # 90th percentile EV - average of top 10% exit velocities
            if len(subset_df) > 0:
                sorted_ev = subset_df['launch_speed'].dropna().sort_values(ascending=False)
                if len(sorted_ev) > 0:
                    top_10_pct = int(max(1, len(sorted_ev) * 0.1))
                    metrics_dict['ev_90th'] = sorted_ev.head(top_10_pct).mean()
                else:
                    metrics_dict['ev_90th'] = None
            else:
                metrics_dict['ev_90th'] = None
        else:
            metrics_dict['avg_ev'] = None
            metrics_dict['max_ev'] = None
            metrics_dict['ev_90th'] = None
        
        # Launch angle metrics
        if 'launch_angle' in subset_df.columns:
            metrics_dict['avg_la'] = subset_df['launch_angle'].mean()
            # Launch Angle Sweet Spot %
            sweet_spot = ((subset_df['launch_angle'] >= 8) & (subset_df['launch_angle'] <= 32)).sum()
            metrics_dict['la_sweet_spot'] = (sweet_spot / len(subset_df)) * 100 if len(subset_df) > 0 else None
        else:
            metrics_dict['avg_la'] = None
            metrics_dict['la_sweet_spot'] = None
        
        # xwOBA
        metrics_dict['xwoba'] = subset_df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in subset_df.columns else None
        
        # ISO - Calculate as SLG - AVG = (Total Bases - Hits) / AB
        # Need to get AB and hits from the full filtered dataset, not just in-play events
        # ISO should be calculated from all at-bats, not just balls in play
        # For now, calculate from in-play events: ISO = (2B + 2*3B + 3*HR) / Total Balls in Play
        # But actually, ISO should be (Total Bases - Hits) / AB for all ABs
        # Since we only have in-play data here, we'll calculate it differently
        # ISO = (Total Bases from in-play) / (Total Balls in Play) - but this isn't quite right
        # Actually, we need to calculate ISO from the full dataset, not just in-play
        # Let's calculate it from events in the subset
        if 'events' in subset_df.columns:
            # Count hits by type
            singles = (subset_df['events'] == 'Single').sum()
            doubles = (subset_df['events'] == 'Double').sum()
            triples = (subset_df['events'] == 'Triple').sum()
            hr_count = (subset_df['events'] == 'Home Run').sum()
            hits = singles + doubles + triples + hr_count
            
            # Calculate total bases
            total_bases = singles + (doubles * 2) + (triples * 3) + (hr_count * 4)
            
            # ISO = (Total Bases - Hits) / AB
            # But we need ABs, not just balls in play
            # For in-play events, we can approximate: ISO ≈ (Total Bases - Hits) / Balls in Play
            # But this isn't quite right. We need the full AB count.
            # Let's set ISO to None here and calculate it properly in offensive metrics
            metrics_dict['iso'] = None  # Will be calculated from full dataset
        else:
            metrics_dict['iso'] = None
        
        # Hard hit, barrel percentages
        if 'launch_speed' in subset_df.columns and len(subset_df) > 0:
            metrics_dict['hard_hit'] = ((subset_df['launch_speed'] >= 95).sum() / len(subset_df)) * 100
        else:
            metrics_dict['hard_hit'] = None
        
        if 'BBECheck' in subset_df.columns and 'is_barrel' in subset_df.columns:
            bbe = subset_df['BBECheck'].sum()
            barrels = subset_df['is_barrel'].sum()
            metrics_dict['barrel'] = (barrels / bbe * 100) if bbe > 0 else None
        else:
            metrics_dict['barrel'] = None
        
        # Batted ball type percentages (GB%, LD%, FB%)
        if 'bb_type' in subset_df.columns and len(subset_df) > 0:
            total_bbe = len(subset_df)
            metrics_dict['gb_pct'] = ((subset_df['bb_type'] == 'ground_ball').sum() / total_bbe) * 100
            metrics_dict['ld_pct'] = ((subset_df['bb_type'] == 'line_drive').sum() / total_bbe) * 100
            metrics_dict['fb_pct'] = ((subset_df['bb_type'] == 'fly_ball').sum() / total_bbe) * 100
        else:
            metrics_dict['gb_pct'] = None
            metrics_dict['ld_pct'] = None
            metrics_dict['fb_pct'] = None
        
        # Pull GB% and Pull AIR% (percentage of all balls in play that are pulled GB/AIR)
        if 'hc_x' in subset_df.columns and 'stand' in subset_df.columns and 'bb_type' in subset_df.columns and len(subset_df) > 0:
            # Get league-wide thresholds
            center_min = thresholds['center_min']
            center_max = thresholds['center_max']
            
            # Filter to only rows with valid hc_x, stand, and bb_type
            valid_data = subset_df[['hc_x', 'stand', 'bb_type']].dropna()
            if len(valid_data) > 0:
                total_bbe = len(valid_data)
                
                # Identify pulled balls based on batter side for each at-bat
                pulled_gb_count = 0
                pulled_air_count = 0
                
                for idx, row in valid_data.iterrows():
                    hc_x_val = row['hc_x']
                    batter_stand = row['stand']
                    bb_type = row['bb_type']
                    
                    # Determine if it's a pulled ball
                    is_pulled = False
                    if batter_stand == 'R':
                        # Right-handed: Pull = left side (low hc_x)
                        if hc_x_val < center_min:
                            is_pulled = True
                    else:  # L
                        # Left-handed: Pull = right side (high hc_x)
                        if hc_x_val > center_max:
                            is_pulled = True
                    
                    # Count pulled groundballs and pulled air balls
                    if is_pulled:
                        if bb_type == 'ground_ball':
                            pulled_gb_count += 1
                        elif bb_type in ['fly_ball', 'line_drive']:
                            pulled_air_count += 1
                
                # Pull GB%: percentage of all balls in play that are pulled groundballs
                metrics_dict['pull_gb_pct'] = (pulled_gb_count / total_bbe) * 100 if total_bbe > 0 else None
                
                # Pull AIR%: percentage of all balls in play that are pulled air balls
                metrics_dict['pull_air_pct'] = (pulled_air_count / total_bbe) * 100 if total_bbe > 0 else None
            else:
                metrics_dict['pull_gb_pct'] = None
                metrics_dict['pull_air_pct'] = None
        else:
            metrics_dict['pull_gb_pct'] = None
            metrics_dict['pull_air_pct'] = None
        
        # Pull/Center/Oppo percentages using hc_x (horizontal location)
        # hc_x: lower values = left side, higher values = right side
        # Use league-wide thresholds and account for batter side for each at-bat (handles switch hitters)
        if 'hc_x' in subset_df.columns and 'stand' in subset_df.columns and len(subset_df) > 0:
            # Filter to only rows with valid hc_x and stand
            valid_data = subset_df[['hc_x', 'stand']].dropna()
            if len(valid_data) > 0:
                total_bbe = len(valid_data)
                
                # Use league-wide thresholds (calculated from full dataset)
                center_min = thresholds['center_min']
                center_max = thresholds['center_max']
                
                # Initialize counters
                pull_count = 0
                center_count = 0
                oppo_count = 0
                
                # Calculate Pull/Center/Oppo for each at-bat based on that at-bat's batter side
                for idx, row in valid_data.iterrows():
                    hc_x_val = row['hc_x']
                    batter_stand = row['stand']
                    
                    # Determine if it's center
                    if center_min <= hc_x_val <= center_max:
                        center_count += 1
                    else:
                        # Determine Pull/Oppo based on batter side for this specific at-bat
                        if batter_stand == 'R':
                            # Right-handed: Pull = left side (low hc_x), Oppo = right side (high hc_x)
                            if hc_x_val < center_min:
                                pull_count += 1
                            else:  # hc_x_val > center_max
                                oppo_count += 1
                        else:  # L
                            # Left-handed: Pull = right side (high hc_x), Oppo = left side (low hc_x)
                            if hc_x_val > center_max:
                                pull_count += 1
                            else:  # hc_x_val < center_min
                                oppo_count += 1
                
                # Calculate percentages
                metrics_dict['pull_pct'] = (pull_count / total_bbe) * 100
                metrics_dict['center_pct'] = (center_count / total_bbe) * 100
                metrics_dict['oppo_pct'] = (oppo_count / total_bbe) * 100
            else:
                metrics_dict['pull_pct'] = None
                metrics_dict['center_pct'] = None
                metrics_dict['oppo_pct'] = None
        else:
            metrics_dict['pull_pct'] = None
            metrics_dict['center_pct'] = None
            metrics_dict['oppo_pct'] = None
        
        return metrics_dict
    
    # Calculate for overall, RHP, and LHP
    overall_metrics = calc_metrics_for_subset(inplay)
    rhp_metrics = calc_metrics_for_subset(inplay_rhp)
    lhp_metrics = calc_metrics_for_subset(inplay_lhp)
    
    # Calculate ISO from full dataset (needs all ABs, not just in-play)
    # ISO = SLG - AVG = (Total Bases - Hits) / AB
    def calc_iso_for_subset(subset_df):
        """Calculate ISO for a subset using all at-bats"""
        if len(subset_df) == 0:
            return None
        
        AB = subset_df['is_ab'].sum() if 'is_ab' in subset_df.columns else 0
        if AB == 0:
            return None
        
        # Count hits by type
        singles = (subset_df['events'] == 'Single').sum() if 'events' in subset_df.columns else 0
        doubles = (subset_df['events'] == 'Double').sum() if 'events' in subset_df.columns else 0
        triples = (subset_df['events'] == 'Triple').sum() if 'events' in subset_df.columns else 0
        hr_count = (subset_df['events'] == 'Home Run').sum() if 'events' in subset_df.columns else 0
        hits = singles + doubles + triples + hr_count
        
        # Calculate total bases
        total_bases = singles + (doubles * 2) + (triples * 3) + (hr_count * 4)
        
        # ISO = (Total Bases - Hits) / AB
        iso = (total_bases - hits) / AB if AB > 0 else None
        return iso
    
    # Calculate ISO for each split
    overall_metrics['iso'] = calc_iso_for_subset(df)
    rhp_metrics['iso'] = calc_iso_for_subset(df_rhp)
    lhp_metrics['iso'] = calc_iso_for_subset(df_lhp)
    
    # Build result dataframe
    result = []
    metric_definitions = [
        ("Exit Velocity (avg)", "avg_ev", 1, "LaunchSpeed", False),
        ("90th Percentile EV", "ev_90th", 1, "EV_90th", False),
        ("Max Exit Velo", "max_ev", 1, "MaxEV", False),
        ("Launch Angle (avg)", "avg_la", 1, "LaunchAngle", False),
        ("Launch Angle Sweet Spot %", "la_sweet_spot", 1, "LASweetSpotPct", False),
        ("xwOBA (In Play)", "xwoba", 3, "xwOBA", False),
        ("ISO", "iso", 3, "ISO", False),
        ("Hard Hit %", "hard_hit", 1, "HardHitPct", False),
        ("Barrel %", "barrel", 1, "BarrelPct", False),
        ("GB%", "gb_pct", 1, "GBPct", True),  # Lower is better
        ("LD%", "ld_pct", 1, "LDPct", False),
        ("FB%", "fb_pct", 1, "FBPct", False),
        ("Pull%", "pull_pct", 1, "PullPct", False),
        ("Center%", "center_pct", 1, "CenterPct", False),
        ("Oppo%", "oppo_pct", 1, "OppoPct", False),
        ("Pull GB%", "pull_gb_pct", 1, "PullGBPct", True),  # Lower is better
        ("Pull AIR%", "pull_air_pct", 1, "PullAIRPct", False),
    ]
    
    for name, key, decimals, ref_key, reverse in metric_definitions:
        overall_val = overall_metrics.get(key)
        rhp_val = rhp_metrics.get(key)
        lhp_val = lhp_metrics.get(key)
        
        # Calculate percentiles
        overall_pct = None
        rhp_pct = None
        lhp_pct = None
        
        if references and ref_key:
            if 'overall' in references and ref_key in references['overall'] and overall_val is not None:
                overall_pct = calculate_percentile(overall_val, references['overall'][ref_key], reverse=reverse)
            if 'rhp' in references and ref_key in references['rhp'] and rhp_val is not None:
                rhp_pct = calculate_percentile(rhp_val, references['rhp'][ref_key], reverse=reverse)
            if 'lhp' in references and ref_key in references['lhp'] and lhp_val is not None:
                lhp_pct = calculate_percentile(lhp_val, references['lhp'][ref_key], reverse=reverse)
        
        # Format values
        if decimals == 1 and name.endswith("%"):
            overall_formatted = format_percent(overall_val, 1) if overall_val is not None else "-"
            rhp_formatted = format_percent(rhp_val, 1) if rhp_val is not None else "-"
            lhp_formatted = format_percent(lhp_val, 1) if lhp_val is not None else "-"
        else:
            overall_formatted = format_metric(overall_val, decimals) if overall_val is not None else "-"
            rhp_formatted = format_metric(rhp_val, decimals) if rhp_val is not None else "-"
            lhp_formatted = format_metric(lhp_val, decimals) if lhp_val is not None else "-"
        
        result.append({
            "Metric": name,
            "Overall": overall_formatted,
            "Overall_Pct": overall_pct,
            "vs RHP": rhp_formatted,
            "vs RHP_Pct": rhp_pct,
            "vs LHP": lhp_formatted,
            "vs LHP_Pct": lhp_pct
        })
    
    return pd.DataFrame(result)

@st.cache_data
def load_bat_path_data():
    """Load Bat_path.csv data"""
    bat_path_file = "Bat_path.csv"
    if os.path.exists(bat_path_file):
        try:
            df = pd.read_csv(bat_path_file, low_memory=False)
            return df
        except Exception as e:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_batting_stance_data():
    """Load and combine batting-stance_LHP.csv and batting-stance_RHP.csv"""
    lhp_file = "batting-stance_LHP.csv"
    rhp_file = "batting-stance_RHP.csv"
    
    combined_df = pd.DataFrame()
    
    # Load LHP data
    if os.path.exists(lhp_file):
        try:
            df_lhp = pd.read_csv(lhp_file, low_memory=False)
            # Normalize names when loading
            df_lhp['name'] = df_lhp['name'].astype(str).str.strip()
            df_lhp['pitch_hand'] = 'L'  # Add pitch_hand column
            combined_df = pd.concat([combined_df, df_lhp], ignore_index=True)
        except Exception as e:
            pass
    
    # Load RHP data
    if os.path.exists(rhp_file):
        try:
            df_rhp = pd.read_csv(rhp_file, low_memory=False)
            # Normalize names when loading
            df_rhp['name'] = df_rhp['name'].astype(str).str.strip()
            df_rhp['pitch_hand'] = 'R'  # Add pitch_hand column
            combined_df = pd.concat([combined_df, df_rhp], ignore_index=True)
        except Exception as e:
            pass
    
    return combined_df

def calculate_bat_path_percentile_references():
    """Calculate percentile references from Bat_path.csv and batting-stance CSVs for all Bat Path metrics"""
    bat_path_df = load_bat_path_data()
    stance_df = load_batting_stance_data()
    
    references = {
        'overall': {},
        'rhp': {},
        'lhp': {}
    }
    
    if len(bat_path_df) == 0 and len(stance_df) == 0:
        return references
    
    # Calculate overall values (average of RHP and LHP for each player)
    # Group by player and calculate overall metrics
    # Check for swing_length column (could be 'swing_length' or 'avg_swing_length')
    swing_length_col = None
    if len(bat_path_df) > 0:
        if 'avg_swing_length' in bat_path_df.columns:
            swing_length_col = 'avg_swing_length'
        elif 'swing_length' in bat_path_df.columns:
            swing_length_col = 'swing_length'
    
    agg_dict = {
        'attack_angle': 'mean',
        'ideal_attack_angle_rate': lambda x: x.mean() * 100,  # Convert to percentage
        'swing_tilt': 'mean',
        'attack_direction': 'mean',
        'avg_bat_speed': 'mean',
        'avg_intercept_y_vs_plate': 'mean',
        'avg_batter_y_position': 'mean',
        'avg_batter_x_position': 'mean'
    }
    if swing_length_col:
        agg_dict[swing_length_col] = 'mean'
    
    # Process Bat_path.csv data
    if len(bat_path_df) > 0:
        player_overall = bat_path_df.groupby('name').agg(agg_dict).reset_index()
        rhp_df = bat_path_df[bat_path_df['pitch_hand'] == 'R'].groupby('name').agg(agg_dict).reset_index()
        lhp_df = bat_path_df[bat_path_df['pitch_hand'] == 'L'].groupby('name').agg(agg_dict).reset_index()
    else:
        player_overall = pd.DataFrame()
        rhp_df = pd.DataFrame()
        lhp_df = pd.DataFrame()
    
    # Process batting-stance data
    if len(stance_df) > 0:
        # Normalize names for consistent matching - make a copy to avoid modifying cached data
        stance_df = stance_df.copy()
        stance_df['name'] = stance_df['name'].astype(str).str.strip()
        stance_agg_dict = {
            'avg_foot_sep': 'mean',
            'avg_stance_angle': 'mean'
        }
        stance_overall = stance_df.groupby('name').agg(stance_agg_dict).reset_index()
        stance_rhp_df = stance_df[stance_df['pitch_hand'] == 'R'].groupby('name').agg(stance_agg_dict).reset_index()
        stance_lhp_df = stance_df[stance_df['pitch_hand'] == 'L'].groupby('name').agg(stance_agg_dict).reset_index()
        
        # Merge stance data with bat_path data if both exist
        if len(player_overall) > 0:
            player_overall = player_overall.merge(stance_overall, on='name', how='outer')
            rhp_df = rhp_df.merge(stance_rhp_df, on='name', how='outer')
            lhp_df = lhp_df.merge(stance_lhp_df, on='name', how='outer')
        else:
            player_overall = stance_overall
            rhp_df = stance_rhp_df
            lhp_df = stance_lhp_df
    
    # Extract reference values (drop NaN)
    # Map column names to reference keys used in metric definitions
    metric_mapping = {
        'attack_angle': 'AttackAngle',
        'swing_tilt': 'SwingTilt',
        'attack_direction': 'attack_direction',  # Store with column name as key
        'avg_bat_speed': 'avg_bat_speed',  # Store with column name as key
        'avg_intercept_y_vs_plate': 'avg_intercept_y_vs_plate',  # Store with column name as key
        'avg_batter_y_position': 'avg_batter_y_position',  # Store with column name as key
        'avg_batter_x_position': 'avg_batter_x_position',  # Store with column name as key
        'ideal_attack_angle_rate': 'ideal_attack_angle_rate',  # Store with column name as key
        'avg_foot_sep': 'avg_foot_sep',
        'avg_stance_angle': 'avg_stance_angle'
    }
    # Add swing_length to mapping if it exists
    if swing_length_col:
        metric_mapping[swing_length_col] = 'SwingLength'
    
    for col, ref_key in metric_mapping.items():
        if col in player_overall.columns:
            # Overall references
            overall_vals = player_overall[col].dropna().values
            if len(overall_vals) > 0:
                if ref_key:
                    references['overall'][ref_key] = overall_vals
                else:
                    references['overall'][col] = overall_vals
            
            # RHP references
            if col in rhp_df.columns:
                rhp_vals = rhp_df[col].dropna().values
                if len(rhp_vals) > 0:
                    if ref_key:
                        references['rhp'][ref_key] = rhp_vals
                    else:
                        references['rhp'][col] = rhp_vals
            
            # LHP references
            if col in lhp_df.columns:
                lhp_vals = lhp_df[col].dropna().values
                if len(lhp_vals) > 0:
                    if ref_key:
                        references['lhp'][ref_key] = lhp_vals
                    else:
                        references['lhp'][col] = lhp_vals
    
    return references

def calculate_bat_path_metrics(df, references=None, batter_name=None):
    """Calculate bat path metrics using Bat_path.csv"""
    if len(df) == 0:
        return pd.DataFrame({
            "Metric": ["No data"], 
            "Overall": ["-"], 
            "vs RHP": ["-"], 
            "vs LHP": ["-"]
        })
    
    # Load Bat_path.csv
    bat_path_df = load_bat_path_data()
    # Don't return early if Bat_path.csv is empty - stance data might still exist
    
    # Get batter name (batter_name is in "Last, First" format)
    if batter_name is None:
        if 'batter_name' in df.columns:
            batter_name = df['batter_name'].iloc[0] if len(df) > 0 else None
        else:
            return pd.DataFrame({
                "Metric": ["No batter name"], 
                "Overall": ["-"], 
                "vs RHP": ["-"], 
                "vs LHP": ["-"]
            })
    
    # Normalize names for matching (strip whitespace) - make a copy to avoid modifying cached data
    bat_path_df = bat_path_df.copy()
    bat_path_df['name'] = bat_path_df['name'].astype(str).str.strip()
    batter_name_normalized = str(batter_name).strip() if batter_name else ""
    
    # Find rows for this batter in Bat_path.csv
    batter_rows = bat_path_df[bat_path_df['name'] == batter_name_normalized]
    
    # Get values for vs RHP (pitch_hand == 'R') and vs LHP (pitch_hand == 'L')
    # If no Bat_path data, use empty DataFrames (stance data might still exist)
    rhp_rows = batter_rows[batter_rows['pitch_hand'] == 'R'] if len(batter_rows) > 0 else pd.DataFrame()
    lhp_rows = batter_rows[batter_rows['pitch_hand'] == 'L'] if len(batter_rows) > 0 else pd.DataFrame()
    
    def get_metric_value(rows, metric_col):
        """Get metric value from rows, averaging if multiple rows"""
        if len(rows) == 0:
            return None
        values = rows[metric_col].dropna()
        if len(values) == 0:
            return None
        return values.mean()
    
    # Calculate Overall as average of RHP and LHP values
    def get_overall_value(rhp_val, lhp_val):
        """Calculate overall value as average of RHP and LHP"""
        if rhp_val is not None and lhp_val is not None:
            return (rhp_val + lhp_val) / 2
        elif rhp_val is not None:
            return rhp_val
        elif lhp_val is not None:
            return lhp_val
        return None
    
    # Get values for each split
    # Attack Angle
    attack_angle_rhp = get_metric_value(rhp_rows, 'attack_angle')
    attack_angle_lhp = get_metric_value(lhp_rows, 'attack_angle')
    attack_angle_overall = get_overall_value(attack_angle_rhp, attack_angle_lhp)
    
    # Ideal Attack Angle % (already a percentage in the CSV as decimal)
    ideal_aa_rhp = get_metric_value(rhp_rows, 'ideal_attack_angle_rate')
    ideal_aa_lhp = get_metric_value(lhp_rows, 'ideal_attack_angle_rate')
    ideal_aa_overall = get_overall_value(ideal_aa_rhp, ideal_aa_lhp)
    # Convert from decimal to percentage
    if ideal_aa_overall is not None:
        ideal_aa_overall = ideal_aa_overall * 100
    if ideal_aa_rhp is not None:
        ideal_aa_rhp = ideal_aa_rhp * 100
    if ideal_aa_lhp is not None:
        ideal_aa_lhp = ideal_aa_lhp * 100
    
    # Swing Path Tilt
    swing_tilt_rhp = get_metric_value(rhp_rows, 'swing_tilt')
    swing_tilt_lhp = get_metric_value(lhp_rows, 'swing_tilt')
    swing_tilt_overall = get_overall_value(swing_tilt_rhp, swing_tilt_lhp)
    
    # Attack Direction
    attack_dir_rhp = get_metric_value(rhp_rows, 'attack_direction')
    attack_dir_lhp = get_metric_value(lhp_rows, 'attack_direction')
    attack_dir_overall = get_overall_value(attack_dir_rhp, attack_dir_lhp)
    
    # Bat Speed (avg)
    bat_speed_rhp = get_metric_value(rhp_rows, 'avg_bat_speed')
    bat_speed_lhp = get_metric_value(lhp_rows, 'avg_bat_speed')
    bat_speed_overall = get_overall_value(bat_speed_rhp, bat_speed_lhp)
    
    # Swing Length - calculate from main dataframe (MLB_data.csv) not Bat_path.csv
    swing_length_col = None
    if 'swing_length' in df.columns:
        swing_length_col = 'swing_length'
        # Split by pitcher handedness
        df_rhp = df[df['p_throws'] == 'R'] if 'p_throws' in df.columns else pd.DataFrame()
        df_lhp = df[df['p_throws'] == 'L'] if 'p_throws' in df.columns else pd.DataFrame()
        
        # Calculate mean swing_length for each split
        swing_length_overall = df['swing_length'].mean() if len(df) > 0 and 'swing_length' in df.columns else None
        swing_length_rhp = df_rhp['swing_length'].mean() if len(df_rhp) > 0 and 'swing_length' in df_rhp.columns else None
        swing_length_lhp = df_lhp['swing_length'].mean() if len(df_lhp) > 0 and 'swing_length' in df_lhp.columns else None
    else:
        swing_length_overall = None
        swing_length_rhp = None
        swing_length_lhp = None
    
    # Intercept vs plate (Depth of Contact)
    intercept_plate_rhp = get_metric_value(rhp_rows, 'avg_intercept_y_vs_plate')
    intercept_plate_lhp = get_metric_value(lhp_rows, 'avg_intercept_y_vs_plate')
    intercept_plate_overall = get_overall_value(intercept_plate_rhp, intercept_plate_lhp)
    
    # Average batter y position (Depth in Box)
    batter_y_rhp = get_metric_value(rhp_rows, 'avg_batter_y_position')
    batter_y_lhp = get_metric_value(lhp_rows, 'avg_batter_y_position')
    batter_y_overall = get_overall_value(batter_y_rhp, batter_y_lhp)
    
    # Average batter x position (Distance off Plate)
    batter_x_rhp = get_metric_value(rhp_rows, 'avg_batter_x_position')
    batter_x_lhp = get_metric_value(lhp_rows, 'avg_batter_x_position')
    batter_x_overall = get_overall_value(batter_x_rhp, batter_x_lhp)
    
    # Load batting-stance data and get stance metrics
    # Note: Names are already normalized in load_batting_stance_data()
    stance_df = load_batting_stance_data()
    foot_sep_rhp = None
    foot_sep_lhp = None
    foot_sep_overall = None
    stance_angle_rhp = None
    stance_angle_lhp = None
    stance_angle_overall = None
    
    if len(stance_df) > 0:
        # Normalize batter_name for matching (names in stance_df are already normalized)
        batter_name_normalized = str(batter_name).strip() if batter_name else ""
        
        # Find rows for this batter in batting-stance data (exact match first)
        stance_rows = stance_df[stance_df['name'] == batter_name_normalized]
        
        # If no exact match, try case-insensitive match as fallback
        if len(stance_rows) == 0:
            stance_rows = stance_df[stance_df['name'].str.lower() == batter_name_normalized.lower()]
        
        if len(stance_rows) > 0:
            stance_rhp_rows = stance_rows[stance_rows['pitch_hand'] == 'R']
            stance_lhp_rows = stance_rows[stance_rows['pitch_hand'] == 'L']
            
            foot_sep_rhp = get_metric_value(stance_rhp_rows, 'avg_foot_sep')
            foot_sep_lhp = get_metric_value(stance_lhp_rows, 'avg_foot_sep')
            foot_sep_overall = get_overall_value(foot_sep_rhp, foot_sep_lhp)
            
            stance_angle_rhp = get_metric_value(stance_rhp_rows, 'avg_stance_angle')
            stance_angle_lhp = get_metric_value(stance_lhp_rows, 'avg_stance_angle')
            stance_angle_overall = get_overall_value(stance_angle_rhp, stance_angle_lhp)
    
    overall_metrics = {
        'attack_angle': attack_angle_overall,
        'ideal_attack_angle_pct': ideal_aa_overall,
        'swing_tilt': swing_tilt_overall,
        'attack_dir': attack_dir_overall,
        'bat_speed': bat_speed_overall,
        'intercept_plate': intercept_plate_overall,
        'batter_y': batter_y_overall,
        'batter_x': batter_x_overall,
        'foot_sep': foot_sep_overall,
        'stance_angle': stance_angle_overall
    }
    rhp_metrics = {
        'attack_angle': attack_angle_rhp,
        'ideal_attack_angle_pct': ideal_aa_rhp,
        'swing_tilt': swing_tilt_rhp,
        'attack_dir': attack_dir_rhp,
        'bat_speed': bat_speed_rhp,
        'intercept_plate': intercept_plate_rhp,
        'batter_y': batter_y_rhp,
        'batter_x': batter_x_rhp,
        'foot_sep': foot_sep_rhp,
        'stance_angle': stance_angle_rhp
    }
    lhp_metrics = {
        'attack_angle': attack_angle_lhp,
        'ideal_attack_angle_pct': ideal_aa_lhp,
        'swing_tilt': swing_tilt_lhp,
        'attack_dir': attack_dir_lhp,
        'bat_speed': bat_speed_lhp,
        'intercept_plate': intercept_plate_lhp,
        'batter_y': batter_y_lhp,
        'batter_x': batter_x_lhp,
        'foot_sep': foot_sep_lhp,
        'stance_angle': stance_angle_lhp
    }
    # Add swing_length to metrics if column exists
    if swing_length_col:
        overall_metrics['swing_length'] = swing_length_overall
        rhp_metrics['swing_length'] = swing_length_rhp
        lhp_metrics['swing_length'] = swing_length_lhp
    
    result = []
    metric_definitions = [
        ("Attack Angle (avg)", "attack_angle", 1, "AttackAngle", False),
        ("Ideal Attack Angle %", "ideal_attack_angle_pct", 1, "ideal_attack_angle_rate", False),
        ("Swing Path Tilt (avg)", "swing_tilt", 1, "SwingTilt", False),
        ("Attack Direction", "attack_dir", None, "attack_direction", False),  # Will use attack_direction as ref_key
        ("Bat Speed (avg)", "bat_speed", 1, "avg_bat_speed", False),  # Will use avg_bat_speed as ref_key
        ("Depth of Contact", "intercept_plate", 2, "avg_intercept_y_vs_plate", False),  # Will use avg_intercept_y_vs_plate as ref_key
        ("Depth in Box", "batter_y", 2, "avg_batter_y_position", False),  # Will use avg_batter_y_position as ref_key
        ("Distance off Plate", "batter_x", 2, "avg_batter_x_position", False),  # Will use avg_batter_x_position as ref_key
        ("Foot Separation", "foot_sep", 2, "avg_foot_sep", False),  # Will use avg_foot_sep as ref_key
        ("Stance Angle", "stance_angle", 1, "avg_stance_angle", False),  # Will use avg_stance_angle as ref_key
    ]
    # Add Swing Length after Bat Speed if column exists
    if swing_length_col:
        # Insert after Bat Speed (index 5, which is after index 4 where Bat Speed is)
        metric_definitions.insert(5, ("Swing Length", "swing_length", 2, "SwingLength", False))  # Use SwingLength as reference key
    
    for name, key, decimals, ref_key, reverse in metric_definitions:
        overall_val = overall_metrics.get(key)
        rhp_val = rhp_metrics.get(key)
        lhp_val = lhp_metrics.get(key)
        
        # Calculate percentiles
        overall_pct = None
        rhp_pct = None
        lhp_pct = None
        
        if references and ref_key:
            if 'overall' in references and ref_key in references['overall'] and overall_val is not None:
                overall_pct = calculate_percentile(overall_val, references['overall'][ref_key], reverse=reverse)
            if 'rhp' in references and ref_key in references['rhp'] and rhp_val is not None:
                rhp_pct = calculate_percentile(rhp_val, references['rhp'][ref_key], reverse=reverse)
            if 'lhp' in references and ref_key in references['lhp'] and lhp_val is not None:
                lhp_pct = calculate_percentile(lhp_val, references['lhp'][ref_key], reverse=reverse)
        
        if name == "Attack Direction" or name == "Stance Angle":
            # Format as degrees if numeric, otherwise as string
            if overall_val is not None and isinstance(overall_val, (int, float)):
                overall_formatted = f"{overall_val:.1f}°"
            else:
                overall_formatted = str(overall_val) if overall_val is not None else "-"
            
            if rhp_val is not None and isinstance(rhp_val, (int, float)):
                rhp_formatted = f"{rhp_val:.1f}°"
            else:
                rhp_formatted = str(rhp_val) if rhp_val is not None else "-"
            
            if lhp_val is not None and isinstance(lhp_val, (int, float)):
                lhp_formatted = f"{lhp_val:.1f}°"
            else:
                lhp_formatted = str(lhp_val) if lhp_val is not None else "-"
        elif name == "Ideal Attack Angle %":
            overall_formatted = format_percent(overall_val, 1) if overall_val is not None else "-"
            rhp_formatted = format_percent(rhp_val, 1) if rhp_val is not None else "-"
            lhp_formatted = format_percent(lhp_val, 1) if lhp_val is not None else "-"
        else:
            overall_formatted = format_metric(overall_val, decimals) if overall_val is not None else "-"
            rhp_formatted = format_metric(rhp_val, decimals) if rhp_val is not None else "-"
            lhp_formatted = format_metric(lhp_val, decimals) if lhp_val is not None else "-"
        
        result.append({
            "Metric": name,
            "Overall": overall_formatted,
            "Overall_Pct": overall_pct,
            "vs RHP": rhp_formatted,
            "vs RHP_Pct": rhp_pct,
            "vs LHP": lhp_formatted,
            "vs LHP_Pct": lhp_pct
        })
    
    return pd.DataFrame(result)

def calculate_contact_metrics(df, references=None):
    """Calculate contact/swing decision metrics with LHP/RHP splits"""
    if len(df) == 0:
        return pd.DataFrame({
            "Metric": ["No data"], 
            "Overall": ["-"], 
            "vs RHP": ["-"], 
            "vs LHP": ["-"]
        })
    
    # Split by pitcher handedness
    df_rhp = df[df['p_throws'] == 'R'] if 'p_throws' in df.columns else pd.DataFrame()
    df_lhp = df[df['p_throws'] == 'L'] if 'p_throws' in df.columns else pd.DataFrame()
    
    def calc_contact_for_subset(subset_df):
        """Calculate contact metrics for a subset"""
        if len(subset_df) == 0:
            return {}
        
        total = len(subset_df)
        swing_pct = (subset_df['is_swing'].sum() / total * 100) if 'is_swing' in subset_df.columns and total > 0 else None
        zone_pct = (subset_df['is_in_zone'].mean() * 100) if 'is_in_zone' in subset_df.columns else None
        
        out_zone = subset_df['is_out_zone'].sum() if 'is_out_zone' in subset_df.columns else 0
        chase_pct = (subset_df['is_chase'].sum() / out_zone * 100) if out_zone > 0 else None
        
        swings = subset_df['is_swing'].sum() if 'is_swing' in subset_df.columns else 0
        whiff_pct = (subset_df['is_whiff'].sum() / swings * 100) if swings > 0 else None
        
        zone_swings = subset_df['is_zone_swing'].sum() if 'is_zone_swing' in subset_df.columns else 0
        zone_whiff = (subset_df['is_zone_whiff'].sum() / zone_swings * 100) if zone_swings > 0 else None
        
        return {
            'swing_pct': swing_pct,
            'zone_pct': zone_pct,
            'chase_pct': chase_pct,
            'whiff_pct': whiff_pct,
            'zone_whiff': zone_whiff
        }
    
    overall_metrics = calc_contact_for_subset(df)
    rhp_metrics = calc_contact_for_subset(df_rhp)
    lhp_metrics = calc_contact_for_subset(df_lhp)
    
    result = []
    metric_definitions = [
        ("Swing %", "swing_pct", "SwingPct", False),
        ("Zone %", "zone_pct", "ZonePct", False),
        ("Chase %", "chase_pct", "ChasePct", True),  # Lower is better
        ("Whiff %", "whiff_pct", "WhiffPct", True),  # Lower is better
        ("Zone Whiff %", "zone_whiff", "ZoneWhiffPct", True),  # Lower is better
    ]
    
    for name, key, ref_key, reverse in metric_definitions:
        overall_val = overall_metrics.get(key)
        rhp_val = rhp_metrics.get(key)
        lhp_val = lhp_metrics.get(key)
        
        # Calculate percentiles
        overall_pct = None
        rhp_pct = None
        lhp_pct = None
        
        if references and ref_key:
            if 'overall' in references and ref_key in references['overall'] and overall_val is not None:
                overall_pct = calculate_percentile(overall_val, references['overall'][ref_key], reverse=reverse)
            if 'rhp' in references and ref_key in references['rhp'] and rhp_val is not None:
                rhp_pct = calculate_percentile(rhp_val, references['rhp'][ref_key], reverse=reverse)
            if 'lhp' in references and ref_key in references['lhp'] and lhp_val is not None:
                lhp_pct = calculate_percentile(lhp_val, references['lhp'][ref_key], reverse=reverse)
        
        overall_formatted = format_percent(overall_val, 1) if overall_val is not None else "-"
        rhp_formatted = format_percent(rhp_val, 1) if rhp_val is not None else "-"
        lhp_formatted = format_percent(lhp_val, 1) if lhp_val is not None else "-"
        
        result.append({
            "Metric": name,
            "Overall": overall_formatted,
            "Overall_Pct": overall_pct,
            "vs RHP": rhp_formatted,
            "vs RHP_Pct": rhp_pct,
            "vs LHP": lhp_formatted,
            "vs LHP_Pct": lhp_pct
        })
    
    return pd.DataFrame(result)

def calculate_quality_metrics(df, references=None):
    """Calculate quality of contact metrics"""
    inplay = df[df['description'] == 'In Play'] if 'description' in df.columns else pd.DataFrame()
    
    if len(inplay) == 0:
        return pd.DataFrame({"Metric": ["No batted-ball events"], "Value": ["-"], "Percentile": [None]})
    
    exit_velo = inplay['launch_speed'].mean() if 'launch_speed' in inplay.columns else None
    max_exit_velo = inplay['launch_speed'].max() if 'launch_speed' in inplay.columns else None
    iso = inplay['iso_value'].mean() if 'iso_value' in inplay.columns else None
    
    hard_hit_pct = None
    if 'launch_speed' in inplay.columns and len(inplay) > 0:
        hard_hit = (inplay['launch_speed'] >= 95).sum()
        hard_hit_pct = (hard_hit / len(inplay)) * 100
    
    weak_pct = None
    if 'launch_speed' in inplay.columns and len(inplay) > 0:
        weak = (inplay['launch_speed'] < 85).sum()
        weak_pct = (weak / len(inplay)) * 100
    
    barrel_pct = None
    if 'BBECheck' in inplay.columns and 'is_barrel' in inplay.columns:
        bbe = inplay['BBECheck'].sum()
        barrels = inplay['is_barrel'].sum()
        barrel_pct = (barrels / bbe * 100) if bbe > 0 else None
    
    metrics = [
        ("Avg Exit Velo", exit_velo, 1, "ExitVelo"),
        ("Max Exit Velo", max_exit_velo, 1, None),
        ("ISO", iso, 3, "ISO"),
        ("Weak %", weak_pct, 1, None),
        ("Hard Hit %", hard_hit_pct, 1, "HardHitPct"),
        ("Barrel %", barrel_pct, 1, "BarrelPct"),
    ]
    
    result = []
    for name, value, decimals, ref_key in metrics:
        if decimals == 1 and name.endswith("%"):
            formatted = format_percent(value, 1) if value is not None else "-"
        else:
            formatted = format_metric(value, decimals) if value is not None else "-"
        percentile = None
        if references and ref_key and ref_key in references and value is not None:
            percentile = calculate_percentile(value, references[ref_key])
        result.append({
            "Metric": name,
            "Value": formatted,
            "Percentile": percentile
        })
    
    return pd.DataFrame(result)

@st.cache_data
def calculate_pitch_tracking_percentile_references(df):
    """Calculate percentile references for pitch tracking metrics by pitch type and split
    Groups by batter and pitch type to get per-batter-per-pitch-type metrics
    Optimized for performance"""
    if len(df) == 0 or 'pitch_name' not in df.columns:
        return {'rhp': {}, 'lhp': {}}
    
    # Normalize pitch names - use vectorized operations for speed
    def normalize_pitch_name(pitch_name):
        if pd.isna(pitch_name):
            return pitch_name
        pitch_str = str(pitch_name).strip()
        if pitch_str == 'Forkball':
            return 'Split-Finger'
        elif pitch_str == 'Knuckle Curve':
            return 'Curveball'
        elif pitch_str == 'Slow Curve':
            return 'Curveball'
        return pitch_str
    
    # Use vectorized operations instead of apply for better performance
    df_filtered = df.copy()
    df_filtered['pitch_name_normalized'] = df_filtered['pitch_name'].apply(normalize_pitch_name)
    excluded_pitches = ['Eephus', 'Other', 'Knuckleball']
    df_filtered = df_filtered[~df_filtered['pitch_name_normalized'].isin(excluded_pitches)].copy()
    
    if len(df_filtered) == 0:
        return {'rhp': {}, 'lhp': {}}
    
    # Split by pitcher handedness
    df_rhp = df_filtered[df_filtered['p_throws'] == 'R'] if 'p_throws' in df_filtered.columns else pd.DataFrame()
    df_lhp = df_filtered[df_filtered['p_throws'] == 'L'] if 'p_throws' in df_filtered.columns else pd.DataFrame()
    
    references = {'rhp': {}, 'lhp': {}}
    
    def calc_references_for_subset(subset_df, subset_name):
        """Calculate references for a subset (RHP or LHP) by grouping by batter and pitch type
        Optimized version using vectorized operations"""
        if len(subset_df) == 0 or 'batter_name' not in subset_df.columns:
            return {}
        
        subset_refs = {}
        
        # Pre-calculate batter totals for usage percentage (more efficient)
        batter_totals = subset_df.groupby('batter_name').size()
        
        # Group by batter and pitch type - this is the expensive operation but necessary
        batter_pitch_groups = subset_df.groupby(['batter_name', 'pitch_name_normalized'])
        
        # Use list comprehensions and vectorized operations where possible
        for (batter_name, pitch_type), pitch_df in batter_pitch_groups:
            if len(pitch_df) == 0:
                continue
            
            # Calculate metrics for this batter-pitch combination
            total_pitches = len(pitch_df)
            batter_total = batter_totals.get(batter_name, 1)
            usage_pct = (total_pitches / batter_total) * 100 if batter_total > 0 else None
            
            # Use vectorized operations for faster calculation
            rv = pitch_df['delta_run_exp'].sum() if 'delta_run_exp' in pitch_df.columns else None
            xwoba = pitch_df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in pitch_df.columns else None
            
            # Filter in-play once
            inplay_mask = pitch_df['description'] == 'In Play' if 'description' in pitch_df.columns else pd.Series([False] * len(pitch_df))
            inplay = pitch_df[inplay_mask] if inplay_mask.any() else pd.DataFrame()
            
            hard_hit_pct = None
            if len(inplay) > 0 and 'launch_speed' in inplay.columns:
                hard_hit = (inplay['launch_speed'] >= 95).sum()
                hard_hit_pct = (hard_hit / len(inplay)) * 100 if len(inplay) > 0 else None
            
            swings = pitch_df['is_swing'].sum() if 'is_swing' in pitch_df.columns else 0
            whiff_pct = None
            if swings > 0:
                whiffs = pitch_df['is_whiff'].sum() if 'is_whiff' in pitch_df.columns else 0
                whiff_pct = (whiffs / swings) * 100
            
            strikeouts = pitch_df['is_strikeout'].sum() if 'is_strikeout' in pitch_df.columns else 0
            k_pct = (strikeouts / total_pitches) * 100 if total_pitches > 0 else None
            
            # Store references by pitch type (aggregate across all batters)
            key_prefix = f"{pitch_type}_"
            
            # Use setdefault for cleaner code
            if usage_pct is not None:
                subset_refs.setdefault(f"{key_prefix}UsagePct", []).append(usage_pct)
            if rv is not None:
                subset_refs.setdefault(f"{key_prefix}RV", []).append(rv)
            if xwoba is not None:
                subset_refs.setdefault(f"{key_prefix}xwOBA", []).append(xwoba)
            if hard_hit_pct is not None:
                subset_refs.setdefault(f"{key_prefix}HardHitPct", []).append(hard_hit_pct)
            if whiff_pct is not None:
                subset_refs.setdefault(f"{key_prefix}WhiffPct", []).append(whiff_pct)
            if k_pct is not None:
                subset_refs.setdefault(f"{key_prefix}KPct", []).append(k_pct)
        
        return subset_refs
    
    references['rhp'] = calc_references_for_subset(df_rhp, 'rhp')
    references['lhp'] = calc_references_for_subset(df_lhp, 'lhp')
    
    return references

def calculate_pitch_tracking(df, references=None):
    """Calculate pitch-by-pitch tracking metrics with Overall, vs RHP, vs LHP splits"""
    if len(df) == 0 or 'pitch_name' not in df.columns:
        return pd.DataFrame()
    
    # Combine pitch types: Forkball -> Split-Finger, Knuckle Curve -> Curveball, Slow Curve -> Curveball
    def normalize_pitch_name(pitch_name):
        """Normalize pitch names by combining similar types"""
        if pd.isna(pitch_name):
            return pitch_name
        pitch_str = str(pitch_name).strip()
        if pitch_str == 'Forkball':
            return 'Split-Finger'
        elif pitch_str == 'Knuckle Curve':
            return 'Curveball'
        elif pitch_str == 'Slow Curve':
            return 'Curveball'
        return pitch_str
    
    # Apply normalization first
    df_filtered = df.copy()
    df_filtered['pitch_name_normalized'] = df_filtered['pitch_name'].apply(normalize_pitch_name)
    
    # Filter out Eephus, Other, and Knuckleball pitch types (after normalization)
    excluded_pitches = ['Eephus', 'Other', 'Knuckleball']
    df_filtered = df_filtered[~df_filtered['pitch_name_normalized'].isin(excluded_pitches)].copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame()
    
    # Split by pitcher handedness (using normalized pitch names)
    df_rhp = df_filtered[df_filtered['p_throws'] == 'R'] if 'p_throws' in df_filtered.columns else pd.DataFrame()
    df_lhp = df_filtered[df_filtered['p_throws'] == 'L'] if 'p_throws' in df_filtered.columns else pd.DataFrame()
    
    def calc_pitch_metrics_for_subset(subset_df, pitch_df):
        """Calculate metrics for a specific pitch type in a subset"""
        if len(pitch_df) == 0:
            return {
                'usage_pct': None,
                'rv': None,
                'xwoba': None,
                'hard_hit_pct': None,
                'whiff_pct': None,
                'k_pct': None
            }
        
        total_pitches = len(pitch_df)
        total_all_pitches = len(subset_df) if len(subset_df) > 0 else 1
        usage_pct = (total_pitches / total_all_pitches) * 100
        
        # Run value
        rv = pitch_df['delta_run_exp'].sum() if 'delta_run_exp' in pitch_df.columns else None
        
        # xwOBA (using estimated_woba_using_speedangle)
        xwoba = pitch_df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in pitch_df.columns else None
        
        # Hard hit %
        inplay = pitch_df[pitch_df['description'] == 'In Play'] if 'description' in pitch_df.columns else pd.DataFrame()
        hard_hit_pct = None
        if len(inplay) > 0 and 'launch_speed' in inplay.columns:
            hard_hit = (inplay['launch_speed'] >= 95).sum()
            hard_hit_pct = (hard_hit / len(inplay)) * 100 if len(inplay) > 0 else None
        
        # Whiff %
        swings = pitch_df['is_swing'].sum() if 'is_swing' in pitch_df.columns else 0
        whiff_pct = None
        if swings > 0:
            whiffs = pitch_df['is_whiff'].sum() if 'is_whiff' in pitch_df.columns else 0
            whiff_pct = (whiffs / swings) * 100
        
        # K%
        strikeouts = pitch_df['is_strikeout'].sum() if 'is_strikeout' in pitch_df.columns else 0
        k_pct = (strikeouts / total_pitches) * 100 if total_pitches > 0 else None
        
        return {
            'usage_pct': usage_pct,
            'rv': rv,
            'xwoba': xwoba,
            'hard_hit_pct': hard_hit_pct,
            'whiff_pct': whiff_pct,
            'k_pct': k_pct
        }
    
    # Get unique normalized pitch types (excluding Eephus and Other)
    pitch_types = sorted([p for p in df_filtered['pitch_name_normalized'].dropna().unique() if p not in excluded_pitches])
    
    if len(pitch_types) == 0:
        return pd.DataFrame()
    
    result = []
    
    for pitch_type in pitch_types:
        # Get pitch data for each split (using normalized pitch name)
        pitch_df_rhp = df_rhp[df_rhp['pitch_name_normalized'] == pitch_type] if len(df_rhp) > 0 else pd.DataFrame()
        pitch_df_lhp = df_lhp[df_lhp['pitch_name_normalized'] == pitch_type] if len(df_lhp) > 0 else pd.DataFrame()
        
        # Calculate metrics for each split (RHP and LHP only, no Overall)
        rhp_metrics = calc_pitch_metrics_for_subset(df_rhp, pitch_df_rhp)
        lhp_metrics = calc_pitch_metrics_for_subset(df_lhp, pitch_df_lhp)
        
        # Calculate percentiles if references provided
        def get_percentile(value, pitch_type, metric_key, split, reverse=False):
            """Get percentile for a pitch tracking metric"""
            if value is None or references is None:
                return None
            ref_key = f"{pitch_type}_{metric_key}"
            if split in references and ref_key in references[split]:
                return calculate_percentile(value, references[split][ref_key], reverse=reverse)
            return None
        
        # Format values and calculate percentiles - RHP and LHP only
        rhp_usage = format_percent(rhp_metrics['usage_pct'], 1) if rhp_metrics['usage_pct'] is not None else "-"
        lhp_usage = format_percent(lhp_metrics['usage_pct'], 1) if lhp_metrics['usage_pct'] is not None else "-"
        rhp_usage_pct = get_percentile(rhp_metrics['usage_pct'], pitch_type, 'UsagePct', 'rhp', reverse=False)
        lhp_usage_pct = get_percentile(lhp_metrics['usage_pct'], pitch_type, 'UsagePct', 'lhp', reverse=False)
        
        rhp_rv = format_metric(rhp_metrics['rv'], 2) if rhp_metrics['rv'] is not None else "-"
        lhp_rv = format_metric(lhp_metrics['rv'], 2) if lhp_metrics['rv'] is not None else "-"
        rhp_rv_pct = get_percentile(rhp_metrics['rv'], pitch_type, 'RV', 'rhp', reverse=False)
        lhp_rv_pct = get_percentile(lhp_metrics['rv'], pitch_type, 'RV', 'lhp', reverse=False)
        
        rhp_xwoba = format_metric(rhp_metrics['xwoba'], 3) if rhp_metrics['xwoba'] is not None else "-"
        lhp_xwoba = format_metric(lhp_metrics['xwoba'], 3) if lhp_metrics['xwoba'] is not None else "-"
        rhp_xwoba_pct = get_percentile(rhp_metrics['xwoba'], pitch_type, 'xwOBA', 'rhp', reverse=False)
        lhp_xwoba_pct = get_percentile(lhp_metrics['xwoba'], pitch_type, 'xwOBA', 'lhp', reverse=False)
        
        rhp_hh = format_percent(rhp_metrics['hard_hit_pct'], 1) if rhp_metrics['hard_hit_pct'] is not None else "-"
        lhp_hh = format_percent(lhp_metrics['hard_hit_pct'], 1) if lhp_metrics['hard_hit_pct'] is not None else "-"
        rhp_hh_pct = get_percentile(rhp_metrics['hard_hit_pct'], pitch_type, 'HardHitPct', 'rhp', reverse=False)
        lhp_hh_pct = get_percentile(lhp_metrics['hard_hit_pct'], pitch_type, 'HardHitPct', 'lhp', reverse=False)
        
        rhp_whiff = format_percent(rhp_metrics['whiff_pct'], 1) if rhp_metrics['whiff_pct'] is not None else "-"
        lhp_whiff = format_percent(lhp_metrics['whiff_pct'], 1) if lhp_metrics['whiff_pct'] is not None else "-"
        rhp_whiff_pct = get_percentile(rhp_metrics['whiff_pct'], pitch_type, 'WhiffPct', 'rhp', reverse=True)  # Lower is better
        lhp_whiff_pct = get_percentile(lhp_metrics['whiff_pct'], pitch_type, 'WhiffPct', 'lhp', reverse=True)  # Lower is better
        
        rhp_k = format_percent(rhp_metrics['k_pct'], 1) if rhp_metrics['k_pct'] is not None else "-"
        lhp_k = format_percent(lhp_metrics['k_pct'], 1) if lhp_metrics['k_pct'] is not None else "-"
        rhp_k_pct = get_percentile(rhp_metrics['k_pct'], pitch_type, 'KPct', 'rhp', reverse=True)  # Lower is better
        lhp_k_pct = get_percentile(lhp_metrics['k_pct'], pitch_type, 'KPct', 'lhp', reverse=True)  # Lower is better
        
        # One row per pitch type with all metrics as columns (including percentiles)
        result.append({
            "Pitch": pitch_type,
            "Usage_RHP": rhp_usage,
            "Usage_RHP_Pct": rhp_usage_pct,
            "Usage_LHP": lhp_usage,
            "Usage_LHP_Pct": lhp_usage_pct,
            "RV_RHP": rhp_rv,
            "RV_RHP_Pct": rhp_rv_pct,
            "RV_LHP": lhp_rv,
            "RV_LHP_Pct": lhp_rv_pct,
            "xwOBA_RHP": rhp_xwoba,
            "xwOBA_RHP_Pct": rhp_xwoba_pct,
            "xwOBA_LHP": lhp_xwoba,
            "xwOBA_LHP_Pct": lhp_xwoba_pct,
            "HH%_RHP": rhp_hh,
            "HH%_RHP_Pct": rhp_hh_pct,
            "HH%_LHP": lhp_hh,
            "HH%_LHP_Pct": lhp_hh_pct,
            "Whiff%_RHP": rhp_whiff,
            "Whiff%_RHP_Pct": rhp_whiff_pct,
            "Whiff%_LHP": lhp_whiff,
            "Whiff%_LHP_Pct": lhp_whiff_pct,
            "K%_RHP": rhp_k,
            "K%_RHP_Pct": rhp_k_pct,
            "K%_LHP": lhp_k,
            "K%_LHP_Pct": lhp_k_pct
        })
    
    return pd.DataFrame(result)

def load_notes():
    """Load notes from file"""
    notes_file = "hitter_profile_notes.json"
    if os.path.exists(notes_file):
        try:
            with open(notes_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_notes(notes):
    """Save notes to file"""
    notes_file = "hitter_profile_notes.json"
    with open(notes_file, 'w') as f:
        json.dump(notes, f, indent=2, default=str)

def generate_download_content(batter_name, filtered_df, references, full_df, batter_name_original, 
                              date_range, offensive_df, bat_path_df, ball_in_play_df, contact_df):
    """Generate PNG and PDF content for download"""
    png_data = None
    pdf_data = None
    
    try:
        # Generate HTML content
        html_content = generate_profile_html(
            batter_name, filtered_df, references, full_df, batter_name_original,
            date_range, offensive_df, bat_path_df, ball_in_play_df, contact_df
        )
        
        # Try Playwright first (best quality)
        try:
            from playwright.sync_api import sync_playwright
            import tempfile
            import os
            import time
            
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    # Set viewport to landscape
                    page.set_viewport_size({"width": 1920, "height": 1080})
                    
                    # Save HTML to temp file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                        f.write(html_content)
                        temp_html = f.name
                    
                    try:
                        # Load the HTML file
                        file_url = f"file:///{temp_html.replace(os.sep, '/')}"
                        page.goto(file_url, wait_until="load", timeout=10000)
                        
                        # Wait for rendering
                        time.sleep(1)
                        
                        # Take full page screenshot
                        screenshot_bytes = page.screenshot(full_page=True, type='png')
                        
                        # Generate PDF
                        pdf_bytes = page.pdf(
                            format='Letter',
                            landscape=True,
                            print_background=True,
                            margin={'top': '0.2in', 'right': '0.2in', 'bottom': '0.2in', 'left': '0.2in'}
                        )
                        
                        png_data = screenshot_bytes
                        pdf_data = pdf_bytes
                        
                    finally:
                        # Clean up temp file
                        try:
                            if os.path.exists(temp_html):
                                os.unlink(temp_html)
                        except:
                            pass
                    
                    browser.close()
                    
                except Exception as playwright_error:
                    # If Playwright fails, fall through to weasyprint
                    st.warning(f"Playwright screenshot failed, using alternative method: {str(playwright_error)}")
                    raise
                    
        except (ImportError, Exception):
            # Fallback to weasyprint for PDF
            try:
                import weasyprint
                from weasyprint import HTML, CSS
                
                css = CSS(string='''
                    @page {
                        size: letter landscape;
                        margin: 0.15in;
                    }
                    body {
                        background: #0e1117;
                        color: white;
                    }
                ''')
                
                pdf_data = HTML(string=html_content).write_pdf(stylesheets=[css])
                # PNG not available with weasyprint fallback
                png_data = None
                
            except Exception as e:
                st.error(f"PDF generation error: {e}")
                import traceback
                st.error(traceback.format_exc())
                pdf_data = None
                png_data = None
                
    except Exception as e:
        st.error(f"Error generating download: {e}")
        import traceback
        st.error(traceback.format_exc())
        png_data = None
        pdf_data = None
    
    return png_data, pdf_data

def generate_profile_html(batter_name, filtered_df, references, full_df, batter_name_original,
                         date_range, offensive_df, bat_path_df, ball_in_play_df, contact_df):
    """Generate HTML representation of the profile matching the app layout exactly"""
    
    # Get date range string
    date_str = ""
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_str = f"Date Range: {date_range[0]} to {date_range[1]}"
    
    # Build left column content
    left_column = ""
    if len(offensive_df) > 0:
        left_column += '<div class="table-section">'
        left_column += '<div class="table-title">Offensive Snapshot</div>'
        left_column += generate_table_html(offensive_df, has_splits=True)
        left_column += '</div>'
    
    if len(bat_path_df) > 0:
        left_column += '<div class="table-section">'
        left_column += '<div class="table-title">Swing Metrics</div>'
        left_column += generate_table_html(bat_path_df, has_splits=True)
        left_column += '</div>'
    
    # Build right column content
    right_column = ""
    if len(ball_in_play_df) > 0:
        right_column += '<div class="table-section">'
        right_column += '<div class="table-title">Ball In Play Metrics</div>'
        right_column += generate_table_html(ball_in_play_df, has_splits=True)
        right_column += '</div>'
    
    if len(contact_df) > 0:
        right_column += '<div class="table-section">'
        right_column += '<div class="table-title">Contact</div>'
        right_column += generate_table_html(contact_df, has_splits=True)
        right_column += '</div>'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{batter_name} - Hitter Profile</title>
    <style>
        @page {{
            size: letter landscape;
            margin: 0.15in;
        }}
        body {{
            font-family: Arial, sans-serif;
            background-color: #ffffff;
            color: #000000;
            padding: 4px;
            font-size: 5.5pt;
            margin: 0;
        }}
        .header {{
            text-align: center;
            margin-bottom: 5px;
        }}
        .header h1 {{
            color: #000000;
            margin: 0;
            font-size: 11pt;
            font-weight: bold;
        }}
        .header .date {{
            color: #666;
            font-size: 5.5pt;
            margin-top: 1px;
        }}
        .main-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}
        .main-table td {{
            width: 50%;
            vertical-align: top;
            padding: 0 2px;
        }}
        .table-section {{
            margin-bottom: 8px;
            page-break-inside: avoid;
        }}
        .table-title {{
            text-align: center;
            color: #000000;
            font-size: 6.5pt;
            font-weight: bold;
            margin-bottom: 2px;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 5pt;
            margin-bottom: 5px;
        }}
        table.data-table th {{
            background-color: transparent;
            color: #000000;
            padding: 1px 2px;
            text-align: center;
            font-size: 5pt;
            font-weight: bold;
        }}
        table.data-table td {{
            padding: 1px 2px;
            text-align: center;
            font-size: 5pt;
        }}
        .metric-name {{
            color: #000000 !important;
            font-weight: bold;
            background-color: transparent !important;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{batter_name}</h1>
        <div class="date">{date_str}</div>
    </div>
    <table class="main-table">
        <tr>
            <td>
                {left_column}
            </td>
            <td>
                {right_column}
            </td>
        </tr>
    </table>
</body>
</html>"""
    return html

def generate_table_html(df, has_splits=False):
    """Generate HTML table with percentile coloring matching the app exactly"""
    html = '<table class="data-table">'
    
    if has_splits and 'Overall' in df.columns:
        # Header
        html += '<thead><tr>'
        html += '<th>Metric</th><th>Overall</th><th>vs RHP</th><th>vs LHP</th>'
        html += '</tr></thead><tbody>'
        
        # Rows
        for idx, row in df.iterrows():
            html += '<tr>'
            # Metric name - white text, no background
            metric_name = str(row["Metric"]).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html += f'<td class="metric-name">{metric_name}</td>'
            
            for col, pct_col in [('Overall', 'Overall_Pct'), ('vs RHP', 'vs RHP_Pct'), ('vs LHP', 'vs LHP_Pct')]:
                value = str(row.get(col, '-'))
                # Escape HTML
                value = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                percentile = row.get(pct_col)
                color = get_color_for_percentile(percentile)
                
                # Determine text color based on background brightness
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    brightness = (r * 299 + g * 587 + b * 114) / 1000
                    text_color = "#000000" if brightness > 128 else "#ffffff"
                else:
                    text_color = "#000000"
                
                # Ensure color is applied with !important to override any CSS
                html += f'<td style="background-color: {color} !important; color: {text_color} !important;">{value}</td>'
            
            html += '</tr>'
    else:
        # Single column table
        html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
        for idx, row in df.iterrows():
            percentile = row.get('Percentile')
            color = get_color_for_percentile(percentile)
            
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                text_color = "#000000" if brightness > 128 else "#ffffff"
            else:
                text_color = "#000000"
            
            metric_name = str(row["Metric"]).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            value = str(row.get("Value", "-")).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            html += '<tr>'
            html += f'<td class="metric-name">{metric_name}</td>'
            html += f'<td style="background-color: {color}; color: {text_color};">{value}</td>'
            html += '</tr>'
    
    html += '</tbody></table>'
    return html

def generate_pdf_with_reportlab(batter_name, filtered_df, references, full_df, batter_name_original,
                                date_range, offensive_df, bat_path_df, ball_in_play_df, contact_df):
    """Generate PDF using reportlab"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    # Title
    story.append(Paragraph(batter_name, title_style))
    story.append(Spacer(1, 10))
    
    # Date range
    if isinstance(date_range, tuple) and len(date_range) == 2:
        story.append(Paragraph(f"Date Range: {date_range[0]} to {date_range[1]}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    # Add tables
    def add_table_to_story(df, title):
        if len(df) > 0:
            story.append(Paragraph(title, styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Prepare table data - only show Metric, Overall, vs RHP, vs LHP columns
            if 'Overall' in df.columns:
                cols_to_show = ['Metric', 'Overall', 'vs RHP', 'vs LHP']
                display_df = df[cols_to_show].copy()
            else:
                display_df = df.copy()
            
            data = [display_df.columns.tolist()]
            for _, row in display_df.iterrows():
                data.append([str(val) for val in row.values])
            
            # Adjust column widths based on number of columns
            col_widths = [2*inch] + [1.5*inch] * (len(display_df.columns) - 1)
            
            table = Table(data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(table)
            story.append(Spacer(1, 10))
    
    add_table_to_story(offensive_df, "Offensive Snapshot")
    add_table_to_story(bat_path_df, "Swing Metrics")
    add_table_to_story(ball_in_play_df, "Ball In Play Metrics")
    add_table_to_story(contact_df, "Contact")
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Main app
def main():
    # Load data (no title, no upload option)
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar filters
    
    # Get unique values and format names - try to find batter name column
    batter_name_col = 'batter_name'
    if 'batter_name' not in df.columns:
        # Try common variations
        possible_names = ['batter', 'Batter', 'BATTER', 'player_name', 'Player', 'name', 'Name']
        for name in possible_names:
            if name in df.columns:
                batter_name_col = name
                # Rename for consistency
                df = df.rename(columns={name: 'batter_name'})
                break
    
    if 'batter_name' in df.columns:
        # Filter out null/empty names
        df = df[df['batter_name'].notna()].copy()
        df = df[df['batter_name'].astype(str).str.strip() != ''].copy()
        
        batter_names_raw = sorted(df['batter_name'].unique())
        # Create display names (First Last) and mapping
        batter_display_names = [format_batter_name(name) for name in batter_names_raw]
        # Create mapping from display name to original name
        name_mapping = {display: original for display, original in zip(batter_display_names, batter_names_raw)}
    else:
        st.error("No batter names found in data")
        st.info(f"Looking for column: 'batter_name' or variations")
        st.info(f"Available columns: {', '.join(df.columns[:30])}")
        st.stop()
    
    if len(batter_display_names) == 0:
        st.error("No batter names found in data after filtering")
        st.stop()
    
    selected_batter_display = st.sidebar.selectbox(
        "Select Batter",
        options=batter_display_names,
        index=0
    )
    
    # Get the original name format for filtering
    selected_batter = name_mapping[selected_batter_display]
    
    # Date range - fix date handling (game_date should exist after load_data processing)
    if 'game_date' in df.columns and len(df) > 0:
        # Filter out invalid dates first
        valid_dates = df[df['game_date'].notna()]['game_date']
        if len(valid_dates) > 0:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            # Ensure dates are reasonable (not 1970)
            if min_date.year < 2000:
                min_date = date(2020, 1, 1)
            if max_date.year < 2000:
                max_date = date.today()
        else:
            min_date = date(2020, 1, 1)
            max_date = date.today()
    else:
        # If no game_date column, use default range (date filtering will be skipped)
        min_date = date(2020, 1, 1)
        max_date = date.today()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Pitcher handedness - removed, always use all
    selected_p_throws = ['R', 'L'] if 'p_throws' in df.columns else []
    
    # Filter data
    filtered_df = df[df['batter_name'] == selected_batter].copy()
    
    # Apply date filter if game_date column exists
    if 'game_date' in filtered_df.columns and isinstance(date_range, tuple) and len(date_range) == 2:
        # Ensure game_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['game_date']):
            try:
                filtered_df['game_date'] = pd.to_datetime(filtered_df['game_date'], errors='coerce')
            except:
                pass
        
        if pd.api.types.is_datetime64_any_dtype(filtered_df['game_date']):
            filtered_df = filtered_df[
                (filtered_df['game_date'].dt.date >= date_range[0]) &
                (filtered_df['game_date'].dt.date <= date_range[1])
            ]
    
    if selected_p_throws and 'p_throws' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['p_throws'].isin(selected_p_throws)]
    
    # Header - use display name, centered, black text
    st.markdown(f"<h3 style='text-align: center; margin-bottom: 0.1rem; margin-top: 0.1rem; color: #000000;'>{selected_batter_display}</h3>", unsafe_allow_html=True)
    
    # Calculate percentile references from full dataset (not filtered)
    references = calculate_percentile_references(df)
    
    # Calculate Bat Path percentile references from Bat_path.csv
    bat_path_references = calculate_bat_path_percentile_references()
    
    # Merge Bat Path references into main references
    for split in ['overall', 'rhp', 'lhp']:
        if split in bat_path_references:
            for key, values in bat_path_references[split].items():
                references[split][key] = values
    
    # Calculate Pitch Tracking percentile references (only if needed - can be slow for large datasets)
    # We'll calculate this lazily when the pitch tracking table is displayed
    # For now, just initialize an empty dict - it will be calculated on demand
    references['pitch_tracking'] = None  # Will be calculated on demand
    
    # Metrics sections - hr line closer to name
    st.markdown("<hr style='margin: 0.1rem 0 0.3rem 0;'>", unsafe_allow_html=True)
    
    # Create columns for metrics - 2 columns now
    col1, col2 = st.columns(2)
    
    def style_dataframe(df_with_percentiles, has_splits=False):
        """Style dataframe with percentile-based colors - white background"""
        if has_splits and 'Overall' in df_with_percentiles.columns:
            # Table with splits (Overall, vs RHP, vs LHP) with percentile coloring
            html = '<table style="width:100%; border-collapse: collapse; font-size: 0.85rem; border: 1px solid #000000;">'
            html += '<thead><tr style="background-color: transparent;"><th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">Metric</th><th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">Overall</th><th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">vs RHP</th><th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">vs LHP</th></tr></thead><tbody>'
            
            for idx, row in df_with_percentiles.iterrows():
                # Add thicker bottom border to separate rows
                html += f'<tr style="border-bottom: 2px solid #000000;"><td style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-bottom: 2px solid #000000; font-weight: bold; background-color: transparent;">{row["Metric"]}</td>'
                
                # Style each value column with percentile coloring (if percentiles exist)
                for col, pct_col in [('Overall', 'Overall_Pct'), ('vs RHP', 'vs RHP_Pct'), ('vs LHP', 'vs LHP_Pct')]:
                    value = row.get(col, '-')
                    # Check if percentile column exists
                    if pct_col in df_with_percentiles.columns:
                        percentile = row.get(pct_col)
                        color = get_color_for_percentile(percentile)
                    else:
                        # No percentiles - use white background
                        color = "#ffffff"
                    
                    # Determine text color based on background brightness
                    if color.startswith('#'):
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                        brightness = (r * 299 + g * 587 + b * 114) / 1000
                        text_color = "#ffffff" if brightness < 128 else "#000000"
                    else:
                        text_color = "#000000"
                    
                    html += f'<td style="padding: 0.2rem 0.3rem; text-align: center; background-color: {color}; color: {text_color}; font-size: 0.8rem; border: 1px solid #000000; border-bottom: 2px solid #000000;">{value}</td>'
                
                html += '</tr>'
            
            html += '</tbody></table>'
            return html
        else:
            # Original single value column table
            styled_df = df_with_percentiles[['Metric', 'Value']].copy()
            
            # Create HTML table with colored cells - white background with black borders
            html = '<table style="width:100%; border-collapse: collapse; font-size: 0.85rem; border: 1px solid #000000;">'
            html += '<thead><tr style="background-color: transparent;"><th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">Metric</th><th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">Value</th></tr></thead><tbody>'
            
            for idx, row in df_with_percentiles.iterrows():
                color = get_color_for_percentile(row.get('Percentile'))
                # Determine text color based on background brightness
                # Calculate brightness (0-255)
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    brightness = (r * 299 + g * 587 + b * 114) / 1000
                    # Use white text if background is dark (brightness < 128), black if light
                    text_color = "#ffffff" if brightness < 128 else "#000000"
                else:
                    text_color = "#000000"
                
                # Add thicker bottom border to separate rows
                html += f'<tr style="border-bottom: 2px solid #000000;"><td style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-bottom: 2px solid #000000; font-weight: bold; background-color: transparent;">{row["Metric"]}</td>'
                html += f'<td style="padding: 0.2rem 0.3rem; text-align: center; background-color: {color}; color: {text_color}; font-size: 0.8rem; border: 1px solid #000000; border-bottom: 2px solid #000000;">{row["Value"]}</td></tr>'
            
            html += '</tbody></table>'
            return html
    
    with col1:
        st.markdown("<h4 style='text-align: center;'>Offensive Snapshot</h4>", unsafe_allow_html=True)
        offensive_df = calculate_offensive_metrics(filtered_df, references)
        st.markdown(style_dataframe(offensive_df, has_splits=True), unsafe_allow_html=True)
        
        st.markdown("<h4 style='text-align: center;'>Swing Metrics</h4>", unsafe_allow_html=True)
        bat_path_df = calculate_bat_path_metrics(filtered_df, references, batter_name=selected_batter)
        st.markdown(style_dataframe(bat_path_df, has_splits=True), unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4 style='text-align: center;'>Ball In Play Metrics</h4>", unsafe_allow_html=True)
        ball_in_play_df = calculate_ball_in_play_metrics(filtered_df, references, full_df=df)
        st.markdown(style_dataframe(ball_in_play_df, has_splits=True), unsafe_allow_html=True)
        
        st.markdown("<h4 style='text-align: center;'>Contact</h4>", unsafe_allow_html=True)
        contact_df = calculate_contact_metrics(filtered_df, references)
        st.markdown(style_dataframe(contact_df, has_splits=True), unsafe_allow_html=True)
    
    # Pitch Tracking Table
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Pitch Tracking</h4>", unsafe_allow_html=True)
    
    # Calculate pitch tracking references on demand (lazy loading for performance)
    if references.get('pitch_tracking') is None:
        with st.spinner("Calculating pitch tracking percentiles..."):
            references['pitch_tracking'] = calculate_pitch_tracking_percentile_references(df)
    
    pitch_tracking_df = calculate_pitch_tracking(filtered_df, references=references.get('pitch_tracking', {}))
    if len(pitch_tracking_df) > 0:
        # Custom styling for pitch tracking table with grouped headers
        html = '<table style="width:100%; border-collapse: collapse; font-size: 0.85rem; border: 1px solid #000000;">'
        
        # First header row with grouped metrics
        html += '<thead>'
        html += '<tr style="background-color: transparent;">'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;" rowspan="2">Pitch</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;" colspan="2">Usage</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;" colspan="2">RV</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;" colspan="2">xwOBA</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;" colspan="2">HH%</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;" colspan="2">Whiff%</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;" colspan="2">K%</th>'
        html += '</tr>'
        
        # Second header row with RHP/LHP - add thicker right borders to LHP columns to separate metric groups
        html += '<tr style="background-color: transparent;">'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">RHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;">LHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">RHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;">LHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">RHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;">LHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">RHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;">LHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">RHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-right: 2px solid #000000; font-weight: bold;">LHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">RHP</th>'
        html += '<th style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; font-weight: bold;">LHP</th>'
        html += '</tr>'
        html += '</thead><tbody>'
        
        # Helper function to get cell style with percentile color
        def get_cell_style(value, percentile):
            if percentile is not None:
                color = get_color_for_percentile(percentile)
                # Determine text color based on background brightness
                if color != "#ffffff":
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    brightness = (r * 299 + g * 587 + b * 114) / 1000
                    text_color = "#ffffff" if brightness < 128 else "#000000"
                else:
                    text_color = "#000000"
            else:
                color = "#ffffff"
                text_color = "#000000"
            return f'padding: 0.2rem 0.3rem; text-align: center; color: {text_color}; font-size: 0.8rem; border: 1px solid #000000; background-color: {color};'
        
        # Data rows with percentile coloring
        for idx, row in pitch_tracking_df.iterrows():
            # Add thicker bottom border to separate rows
            html += '<tr style="border-bottom: 2px solid #000000;">'
            html += f'<td style="padding: 0.2rem 0.3rem; text-align: center; color: #000000; font-size: 0.8rem; border: 1px solid #000000; border-bottom: 2px solid #000000; font-weight: bold; background-color: transparent;">{row["Pitch"]}</td>'
            
            # Usage - add thicker right border to LHP column to separate from RV
            usage_rhp_style = get_cell_style(row["Usage_RHP"], row.get("Usage_RHP_Pct")) + " border-bottom: 2px solid #000000;"
            usage_lhp_style = get_cell_style(row["Usage_LHP"], row.get("Usage_LHP_Pct")) + " border-bottom: 2px solid #000000; border-right: 2px solid #000000;"
            html += f'<td style="{usage_rhp_style}">{row["Usage_RHP"]}</td>'
            html += f'<td style="{usage_lhp_style}">{row["Usage_LHP"]}</td>'
            # RV - add thicker right border to LHP column to separate from xwOBA
            rv_rhp_style = get_cell_style(row["RV_RHP"], row.get("RV_RHP_Pct")) + " border-bottom: 2px solid #000000;"
            rv_lhp_style = get_cell_style(row["RV_LHP"], row.get("RV_LHP_Pct")) + " border-bottom: 2px solid #000000; border-right: 2px solid #000000;"
            html += f'<td style="{rv_rhp_style}">{row["RV_RHP"]}</td>'
            html += f'<td style="{rv_lhp_style}">{row["RV_LHP"]}</td>'
            # xwOBA - add thicker right border to LHP column to separate from HH%
            xwoba_rhp_style = get_cell_style(row["xwOBA_RHP"], row.get("xwOBA_RHP_Pct")) + " border-bottom: 2px solid #000000;"
            xwoba_lhp_style = get_cell_style(row["xwOBA_LHP"], row.get("xwOBA_LHP_Pct")) + " border-bottom: 2px solid #000000; border-right: 2px solid #000000;"
            html += f'<td style="{xwoba_rhp_style}">{row["xwOBA_RHP"]}</td>'
            html += f'<td style="{xwoba_lhp_style}">{row["xwOBA_LHP"]}</td>'
            # HH% - add thicker right border to LHP column to separate from Whiff%
            hh_rhp_style = get_cell_style(row["HH%_RHP"], row.get("HH%_RHP_Pct")) + " border-bottom: 2px solid #000000;"
            hh_lhp_style = get_cell_style(row["HH%_LHP"], row.get("HH%_LHP_Pct")) + " border-bottom: 2px solid #000000; border-right: 2px solid #000000;"
            html += f'<td style="{hh_rhp_style}">{row["HH%_RHP"]}</td>'
            html += f'<td style="{hh_lhp_style}">{row["HH%_LHP"]}</td>'
            # Whiff% - add thicker right border to LHP column to separate from K%
            whiff_rhp_style = get_cell_style(row["Whiff%_RHP"], row.get("Whiff%_RHP_Pct")) + " border-bottom: 2px solid #000000;"
            whiff_lhp_style = get_cell_style(row["Whiff%_LHP"], row.get("Whiff%_LHP_Pct")) + " border-bottom: 2px solid #000000; border-right: 2px solid #000000;"
            html += f'<td style="{whiff_rhp_style}">{row["Whiff%_RHP"]}</td>'
            html += f'<td style="{whiff_lhp_style}">{row["Whiff%_LHP"]}</td>'
            # K% - no right border needed (last metric group)
            k_rhp_style = get_cell_style(row["K%_RHP"], row.get("K%_RHP_Pct")) + " border-bottom: 2px solid #000000;"
            k_lhp_style = get_cell_style(row["K%_LHP"], row.get("K%_LHP_Pct")) + " border-bottom: 2px solid #000000;"
            html += f'<td style="{k_rhp_style}">{row["K%_RHP"]}</td>'
            html += f'<td style="{k_lhp_style}">{row["K%_LHP"]}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No pitch tracking data available.")
    
    # Notes Archive (moved here, before Hitter Notes)
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### Notes Archive")
    notes = load_notes()
    if notes:
        notes_list = []
        for batter, note in notes.items():
            if note.strip():  # Only show non-empty notes
                # Format batter name for display
                display_name = format_batter_name(batter)
                notes_list.append({"Batter": display_name, "Note": note[:100] + "..." if len(note) > 100 else note})
        if notes_list:
            notes_df = pd.DataFrame(notes_list)
            st.dataframe(notes_df, use_container_width=True, hide_index=True)
        else:
            st.info("No notes saved yet.")
    else:
        st.info("No notes saved yet.")
    
    # Additional visualizations can be added here
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### Additional Visualizations")
    
    # Add vertical space for visualization section
    st.markdown("<div style='min-height: 300px;'></div>", unsafe_allow_html=True)
    
    # Hitter Notes section (moved here, after Additional Visualizations)
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    
    # Load notes - use original name format for lookup
    current_note = notes.get(selected_batter, "")
    
    note_text = st.text_area(
        "Notes",
        value=current_note,
        height=75
    )
    
    # Center the save button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 Save Notes", use_container_width=True):
            # Save using original name format
            notes[selected_batter] = note_text
            save_notes(notes)
            st.session_state.profile_notes = notes

if __name__ == "__main__":
    main()


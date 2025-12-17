"""
Data processing utilities for Hitter Profiles
"""
import pandas as pd
import numpy as np

# Event definitions
SWING_EVENTS = [
    "Foul Tip", "Swinging Strike", "Swinging Strike (blocked)",
    "Missed Bunt", "Foul", "In Play", "Foul Bunt", "bunt_foul_tip"
]

STRIKE_EVENTS = SWING_EVENTS + ["Called Strike"]

WHIFF_EVENTS = [
    "Swinging Strike", "Foul Tip", "Foul Bunt",
    "Missed Bunt", "Swinging Strike (blocked)"
]

HIT_EVENTS = ["Single", "Double", "Triple", "Home Run"]

PA_EVENTS = [
    "Out", "FC", "Force", "Home Run", "GiDP", "Error", "Single", "Double", "Triple",
    "FC Out", "DP", "Other", "strikeout_double_play", "Strikeout", "Walk",
    "Sac Fly", "Sac Bunt", "Sac Fly DP", "HBP", "catcher_interf"
]

AB_EVENTS = [
    "Out", "FC", "Force", "Home Run", "GiDP", "Error", "Single", "Double", "Triple",
    "FC Out", "DP", "Other", "Strikeout", "strikeout_double_play"
]

HR_CHECK = ["Home Run"]
XBH_CHECK = ["Home Run", "Double", "Triple"]

def preprocess_data(df):
    """
    Preprocess the dataframe to add derived columns
    Similar to the R preprocessing
    """
    df = df.copy()
    
    # Convert description and events to match expected values
    if 'description' in df.columns:
        df['description'] = df['description'].replace({
            "foul_tip": "Foul Tip",
            "hit_into_play": "In Play",
            "foul": "Foul",
            "swinging_strike": "Swinging Strike",
            "called_strike": "Called Strike",
            "blocked_ball": "Blocked Ball",
            "ball": "Ball",
            "swinging_strike_blocked": "Swinging Strike (blocked)",
            "hit_by_pitch": "HBP",
            "missed_bunt": "Missed Bunt",
            "foul_bunt": "Foul Bunt"
        })
    
    if 'events' in df.columns:
        df['events'] = df['events'].replace({
            "field_out": "Out",
            "fielders_choice": "FC",
            "force_out": "Force",
            "home_run": "Home Run",
            "grounded_into_double_play": "GiDP",
            "field_error": "Error",
            "strikeout": "Strikeout",
            "walk": "Walk",
            "single": "Single",
            "double": "Double",
            "triple": "Triple",
            "fielders_choice_out": "FC Out",
            "double_play": "DP",
            "sac_bunt": "Sac Bunt",
            "other_out": "Other",
            "sac_fly_double_play": "Sac Fly DP",
            "sac_fly": "Sac Fly",
            "hit_by_pitch": "HBP",
            "catcher_interf": "C Interference",
            "caught_stealing_2b": "CS 2B",
            "strikeout_double_play": "K DP"
        })
        df['events'] = df['events'].fillna("Not In Play")
    
    # Calculate derived binary indicators
    if 'description' in df.columns:
        df['is_swing'] = df['description'].isin(SWING_EVENTS).astype(int)
        df['is_whiff'] = df['description'].isin(WHIFF_EVENTS).astype(int)
        df['is_contact'] = df['description'].isin(["In Play", "Foul", "foul_pitchout"]).astype(int)
    else:
        df['is_swing'] = 0
        df['is_whiff'] = 0
        df['is_contact'] = 0
    
    if 'zone' in df.columns:
        df['is_in_zone'] = df['zone'].isin(range(1, 10)).astype(int)
        df['is_out_zone'] = 1 - df['is_in_zone']
    else:
        df['is_in_zone'] = 0
        df['is_out_zone'] = 1
    
    df['is_chase'] = df['is_swing'] * df['is_out_zone']
    df['is_zone_whiff'] = (df['is_whiff'] & df['is_in_zone']).astype(int)
    df['is_zone_swing'] = (df['is_swing'] & df['is_in_zone']).astype(int)
    
    if 'events' in df.columns:
        df['is_strikeout'] = (df['events'] == "Strikeout").astype(int)
        df['is_ab'] = df['events'].isin(AB_EVENTS).astype(int)
        df['is_hit'] = df['events'].isin(HIT_EVENTS).astype(int)
        df['is_PA'] = df['events'].isin(PA_EVENTS).astype(int)
        df['is_HR'] = df['events'].isin(HR_CHECK).astype(int)
        df['is_XBH'] = df['events'].isin(XBH_CHECK).astype(int)
    else:
        df['is_strikeout'] = 0
        df['is_ab'] = 0
        df['is_hit'] = 0
        df['is_PA'] = 0
        df['is_HR'] = 0
        df['is_XBH'] = 0
    
    if 'pitch_number' in df.columns:
        df['is_first_pitch'] = (df['pitch_number'] == 1).astype(bool)
    
    if 'launch_speed' in df.columns:
        df['is_hard_hit'] = ((df['launch_speed'] >= 95) & (df['launch_speed'] <= 130)).astype(int)
        df['BBECheck'] = (df['description'] == "In Play").astype(int) if 'description' in df.columns else 0
    else:
        df['is_hard_hit'] = 0
        df['BBECheck'] = 0
    
    # Calculate barrel
    if 'launch_speed' in df.columns and 'launch_angle' in df.columns and 'description' in df.columns:
        in_play = df['description'] == "In Play"
        df['is_barrel'] = False
        
        # Barrel conditions
        conditions = [
            (in_play & (df['launch_speed'] >= 98) & (df['launch_speed'] <= 100) & 
             (df['launch_angle'] >= 26) & (df['launch_angle'] <= 30)),
            (in_play & (df['launch_speed'] > 100) & (df['launch_speed'] <= 102) & 
             (df['launch_angle'] >= 24) & (df['launch_angle'] <= 33)),
            (in_play & (df['launch_speed'] > 102) & (df['launch_speed'] <= 104) & 
             (df['launch_angle'] >= 22) & (df['launch_angle'] <= 36)),
            (in_play & (df['launch_speed'] > 104) & 
             (df['launch_angle'] >= 20) & (df['launch_angle'] <= 38))
        ]
        
        df['is_barrel'] = np.any(conditions, axis=0).astype(int)
    else:
        df['is_barrel'] = 0
    
    # 0-0 count indicators
    if 'balls' in df.columns and 'strikes' in df.columns:
        df['is_zero_zero'] = ((df['balls'] == 0) & (df['strikes'] == 0)).astype(int)
        df['zero_zero_swing'] = ((df['balls'] == 0) & (df['strikes'] == 0) & (df['is_swing'] == 1)).astype(int)
        df['pre2k'] = (df['strikes'] < 2).astype(int)
        df['pre2k_whiff'] = ((df['strikes'] < 2) & (df['is_whiff'] == 1)).astype(int)
    else:
        df['is_zero_zero'] = 0
        df['zero_zero_swing'] = 0
        df['pre2k'] = 0
        df['pre2k_whiff'] = 0
    
    return df





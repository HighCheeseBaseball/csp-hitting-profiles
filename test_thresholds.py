import pandas as pd
from data_utils import preprocess_data

df = pd.read_csv('MLB_data.csv', nrows=100000)
df = preprocess_data(df)
abrams = df[(df['batter_name'].str.contains('Abrams', case=False, na=False)) & (df['description'] == 'In Play')]
hc_x = abrams['hc_x'].dropna()

print('Testing different thresholds to match Savant (43.8% pull, 34.6% center, 21.6% oppo):')
print()

for cmin in [118, 119, 120, 121, 122]:
    for cmax in [152, 153, 154, 155, 156]:
        pull = (hc_x > cmax).sum() / len(hc_x) * 100
        center = ((hc_x >= cmin) & (hc_x <= cmax)).sum() / len(hc_x) * 100
        oppo = (hc_x < cmin).sum() / len(hc_x) * 100
        
        pull_diff = abs(pull - 43.8)
        center_diff = abs(center - 34.6)
        oppo_diff = abs(oppo - 21.6)
        
        if pull_diff < 5 and center_diff < 5 and oppo_diff < 5:
            print(f'Center {cmin}-{cmax}: Pull={pull:.1f}% (diff: {pull_diff:.1f}), Center={center:.1f}% (diff: {center_diff:.1f}), Oppo={oppo:.1f}% (diff: {oppo_diff:.1f})')


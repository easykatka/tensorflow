

import pandas as pd
df = pd.read_pickle('dataframe_ver_1.pkl')

categories = {}
df = df.sample(frac=1).reset_index(drop=True)


for key,value in enumerate(df['parentid'].unique()):
    categories[value] = key + 1


print(categories)
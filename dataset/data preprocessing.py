import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data_raw = pd.read_csv("dataset/merged_df_weather_sub_base2.csv")

# Remove data features 
#DAS_features = [col for col in data_raw.columns if "DAS" in col]
local = ['lat','lon', 'year']
# columns_to_remove = ['planting_doy', 'Sw_ratio1mPlanting', 'soc30cm','soc60cm',
#                       'ResidueDMatPlanting','SummerAvgt30cmN03','SummerColdD','SummerDnit',
#                       'AnnualIrrigTimes','SummerWT30cm']


# Create the OneHotEncoder object
encoder = OneHotEncoder()

# Fit the encoder to the categories and transform the data
one_hot = encoder.fit_transform(np.array(data_raw["cultivar"]).reshape(-1, 1))

#columns_to_remove += DAS_features + local
data = data_raw.drop(columns=local)
data = data.drop(data.columns[0], axis=1)
#data["one_hot"] = list(one_hot.toarray()

df_cleaned = data.dropna()

value_counts = df_cleaned["cultivar"].value_counts()
print("Data Sample for each cultivar",value_counts )

# # Define a function to filter rows within each group
# def custom_filter(group):
#     unique_cols_except_last = group.iloc[:, :-1].apply(lambda col: col.nunique() == 1)
#     if unique_cols_except_last.all():
#         return group
#     if group['yield_kg_ha'].nunique() == 1 and not group.iloc[:, :-1].duplicated().any():
#         return group

# # Apply the filtering function on each group and concatenate the results
# filtered_data = pd.concat([custom_filter(group) for _, group in df_cleaned.groupby('cultivar', group_keys=False)])


df_cleaned.to_csv('dataset/clean_yield.csv', index=False)
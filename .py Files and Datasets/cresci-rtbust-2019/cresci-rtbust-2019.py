# Import pandas to read .tsv and prepare dataset
import pandas as pd
tsvdata = pd.read_csv('cresci-rtbust-2019.tsv', sep='\t', header=None, names=['id', 'label']) 
labels = tsvdata[['id', 'label']]     

# Import json to read json file
import json
with open('cresci-rtbust-2019_tweets.json') as f:
    jsondata = json.load(f)

# Flatten data contained within json
from flatten_json import flatten
flattened_data = [flatten(d) for d in jsondata]
df = pd.DataFrame(flattened_data)

# Merge .tsv and .json files into one
merged_data = pd.merge(df, labels, left_on='user_id', right_on='id')
merged_data.drop(['id', 'user_id'], axis=1, inplace=True)

# Import numpy to split newly merged data
import numpy as np
data = np.array(merged_data)
features = data[0: , :-1]
labels = data[0: ,-1]

# Create encoder then encode labels
from sklearn.preprocessing import LabelEncoder
Lencoder = LabelEncoder()
encLabels = Lencoder.fit_transform(labels)

# Create encoder then encode features
from sklearn.preprocessing import OneHotEncoder
Fencoder = OneHotEncoder()
encFeat = Fencoder.fit_transform(features)

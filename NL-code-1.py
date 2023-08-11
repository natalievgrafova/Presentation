# %%
import pandas as pd

# %%
from textblob import TextBlob 
from textblob import Blobber
from textblob_nl import PatternTagger, PatternAnalyzer
tb_fr = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())


# %%
df = pd.read_csv('dutch_gegevens2.csv')

# %%
df = df[['title', 'text', 'date', 'url', '_id']]


# %%
key_words_data = ["persoonsgegevens", "gegevensgrondslag", "vertrouwelijkheid", "delen van gegevens", "toegang tot gegevens", "hergebruik van gegevens", "gebruik van gegevens", "gegevensanalyse", "gebruik van gegevens", "gegevensbescherming", "data", "data use" ,"data reusability", "data reuse", "data sharing", "data access", "data privacy", "data protection"]

# %%
df

# %%
df.iloc[2,1]

# %%
df.head(2)

# %%
len(df)

# %%
list_data = []
for row in range(len(df)):
    if any(word in str(df.iloc[row, 1]) for word in key_words_data):
        if str(df.iloc[row, 1]) not in list_data:
            list_data.append(str(df.iloc[row, 1]))

# %%
len(list_data)

# %%
list_data[1]

# %%
df = df[df['text'].isin(list_data)]

# %%
df = df.drop_duplicates(subset=['text'])

# %%
df

# %%
def get_polarity_fr(doc):
        polarity = tb_fr(doc).sentiment[0]
        return polarity

# %%
polarity_list = []
for row in range(len(df)): 
    polarity_list.append(round(get_polarity_fr(str(df.iloc[row,1])),5))
    

# %%
polarity_list[:4]

# %%
polarity_list[:10]

# %%
df['polarity'] = polarity_list

# %%
df.head(10)

# %%
df['language'] = 'nl'

# %%
df

# %%
import re

# %%
def change_year(column):
    def find_year(s: str):
            matcher = re.search('\d\d\d\d',str(s))
            
            return int(matcher.group(0))
            
    column = column.apply(find_year)
    return column

# %%
df['date'] = change_year(df['date'])

# %%
df['date'].unique()

# %% [markdown]
# Graphs

# %%




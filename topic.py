import pandas as pd

df_item = pd.read_csv("items.csv", sep="|")

df_item.itemID = df_item.itemID.astype(str)

subtopic = df_item.subtopics
subtopic = subtopic.str.strip('][')
subtopic = subtopic.str.split(',', expand = True)
subtopic = subtopic.add_prefix('subtopic_')
df_item = pd.concat([df_item, subtopic], axis=1)
df_item.head()
df_item.shape[0]
df_item['main topic'].fillna(df_item['subtopic_1'], inplace=True)
df_item['main topic'].isna().sum()
df_item[(df_item['main topic'].isna()) & (df_item['subtopic_1'].isna())]
df_item['main topic'].fillna('', inplace = True)
df_item[df_item['subtopic_1'].str.startswith("Y")&~(df_item['main topic'].str.startswith("Y"))]
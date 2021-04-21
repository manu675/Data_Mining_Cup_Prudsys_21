import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re

item_dt = pd.read_csv("items.csv", sep='|')
trans_dt = pd.read_csv("transactions.csv", sep='|')
eval_dt = pd.read_csv("evaluation.csv")
eval_dt.itemID.isin([12])

item_dt.head()
item_dt.columns
item_dt.shape
item_dt.author.value_counts().head(20)
item_dt.publisher.unique().shape[0]
item_dt.title.unique().shape[0]
item_dt['main topic'].unique().shape[0]
item_dt.publisher.value_counts().head(20)
item_dt.publisher.unique().shape
item_dt.isna().sum()
## How to deal with NA values.
child = {True:"Child", False: "Adult"}
item_dt['main topic'].fillna("", inplace=True)
item_dt['author'].fillna("Unknown", inplace=True)

#Exploration for item_dt
item_dt["target"] = [child[t] for t in list(item_dt['main topic'].str.contains('Y'))]
item_dt.target.value_counts()
item_dt["fiction"] = item_dt['main topic'].str.contains('F')
item_dt.fiction.value_counts()
item_dt['itemID']=item_dt['itemID'].astype(str)
item_dt[item_dt['author'].str.contains(',')]['author'].head()
item_dt.groupby(['author','main topic']).count()

#Should calculate percent that customer would by the author's book.
#Maybe publisher can be also meaningful.
#Should make category to distinguish child's and adult's.

#EDA Trans_dt
trans_dt.head()
trans_dt.info()
trans_dt.describe()
trans_dt.columns
trans_dt['sessionID'] = trans_dt['sessionID'].astype(str)
trans_dt['itemID'] = trans_dt['itemID'].astype(str)
trans_dt['order'].unique().shape[0]

item_count = trans_dt.groupby('itemID').sessionID.count().value_counts().sort_index()
item_count/sum(item_count)
(item_count/sum(item_count)).head(20).plot.bar()
plt.xticks(rotation=45)
plt.title("Count for specific items")
plt.show()

item_click = trans_dt.groupby('itemID').click.sum().value_counts().sort_index()
item_click/sum(item_click)
(item_click/sum(item_click)).head(20).plot.bar()
plt.xticks(rotation=45)
plt.title("Click for specific items")
plt.show()

item_order = trans_dt.groupby('itemID').order.sum().value_counts().sort_index()
item_order/sum(item_order)
(item_order/sum(item_order)).head(20).plot.bar()
plt.xticks(rotation=45)
plt.title("Order for specific items")
plt.show()

eval_dt['itemID'] = eval_dt['itemID'].astype(str)
eval_dt.shape
eval_dt[eval_dt['itemID'].isin(trans_dt['itemID'])].shape[0]
trans_dt.order.isna().sum()
trans_dt.basket.value_counts().plot.bar()
plt.show()
trans_dt.click.value_counts()
trans_dt.groupby('sessionID')['order'].sum().value_counts().sort_index().plot.bar()
plt.show()
trans_dt.groupby('sessionID')['click'].sum().value_counts()[0:30].plot.bar()
plt.show()

#Should make discipline with click and order.
#Should think about transaction without click and order.

zero_click_dt = trans_dt[trans_dt['click']==0]
zero_click_dt.groupby('sessionID').itemID.count().plot.bar()
plt.show()
zero_click_dt.groupby('sessionID').order.sum().value_counts().plot.bar()
plt.xlabel("Number of Orders when zero click happened")
plt.show()

zero_order_dt = trans_dt[trans_dt['order']==0]
zero_order_dt.groupby('sessionID').itemID.count().plot.bar()
plt.show()

# How should we handle the click without order?
# Is it negative or positive?
zero_order_dt.groupby('sessionID').click.sum().value_counts().head(20).plot.bar()
plt.xlabel("Number of clicks when zero order happened")
plt.show()

# How about filter row without any click and transaction?
zero_click_order_dt = trans_dt[(trans_dt['order']==0)&(trans_dt['click']==0)]
zero_click_order_dt.shape

trans_dt['Number in the Session'] = trans_dt.groupby('sessionID').itemID.transform('count')
trans_dt['Number in the Session'].value_counts()
trans_dt['orders in session'] = trans_dt.groupby('sessionID').order.transform('sum')
trans_dt['clicks in session'] = trans_dt.groupby('sessionID').click.transform('sum')
trans_dt.describe()
trans_dt.plot.scatter(x='clicks in session', y = 'orders in session')
plt.show()

combined_dt = item_dt.merge(trans_dt, on='itemID', how="left")
combined_dt.isna().sum()
## Dropped NA value to deal with data only appeared in items dt.
combined_dt = combined_dt.dropna()
combined_dt.shape[0]

## Dealing with author and transaction.
session_count = combined_dt.groupby('sessionID').count().itemID.value_counts().sort_index()
(session_count/sum(session_count)).plot.bar()
session_count/sum(session_count)
plt.title("Numbers of transactions in a session")
plt.xticks(rotation=45)
plt.show()

author_count = combined_dt.groupby(["author","sessionID"]).itemID.count().value_counts().sort_index()
(author_count/sum(author_count)).plot.bar()
author_count/sum(author_count)
plt.title("An author in same session")
plt.show()

combined_dt["author_cooccurence"] = combined_dt.groupby(["author","sessionID"]).itemID.transform("count")
combined_dt["author_cooccurence"].value_counts()

combined_dt["session_author_order"] = combined_dt.groupby(["author","sessionID"]).order.transform("sum")
combined_dt["session_author_click"] = combined_dt.groupby(["author","sessionID"]).click.transform("sum")

combined_dt.groupby(["author","sessionID"]).click.sum().value_counts().sort_index()/combined_dt.shape[0]
combined_dt.groupby(["author","sessionID"]).click.sum().value_counts().sort_index().plot.bar()
plt.title("Clicks in a session for an author.")
plt.tick_params('x', bottom=False, top=False, labelbottom=False)
plt.ylim([0,30000])
plt.show()

combined_dt.groupby(["author","sessionID"]).order.sum().value_counts().sort_index()/combined_dt.shape[0]
combined_dt.groupby(["author","sessionID"]).order.sum().value_counts().sort_index().plot.bar()
plt.title("Orders in a session for an author.")
plt.tick_params('x', bottom=False, top=False, labelbottom=False)
plt.ylim([0,15000])
plt.show()

combined_dt.shape
combined_dt[combined_dt['session_author_order']>1].shape[0]/combined_dt.shape[0]

#Deal with authors in evaluation
eval_dt['itemID'] = eval_dt.itemID.astype(str)
item_eval_dt = item_dt.merge(eval_dt, on="itemID")
#Make dataframe from item_dt which has same author with item in eval_dt
author_eval_dt = item_dt[item_dt["author"].isin(item_eval_dt["author"])]
author_eval_dt.shape
author_eval_dt = author_eval_dt[author_eval_dt['author']!='Unknown']
eval_author_count = author_eval_dt.groupby("author").count().itemID
eval_author_count[eval_author_count==1479]
eval_author_count.value_counts().sort_index()
eval_author_count.value_counts().sort_index().plot.bar()
plt.title("Number of books written by authors in evaluation data")
plt.tick_params('x', bottom=False, top=False, labelbottom=False)
plt.show()

##Dealing with topic and session.
plt.style.use('ggplot')
topic_count = combined_dt.groupby(["main topic","sessionID"]).itemID.count().value_counts().sort_index()
(topic_count/sum(topic_count)).plot.bar()
topic_count/sum(topic_count)
plt.title("Main Topic in same session")
plt.text(0, 0.95, '92.29%')
plt.xticks(rotation=45)
plt.show()

topic_click = combined_dt.groupby(["main topic","sessionID"]).click.sum().value_counts().sort_index()
topic_click/sum(topic_click)
combined_dt.groupby(["main topic","sessionID"]).click.sum().value_counts().sort_index().plot.bar()
plt.title("Clicks in a session for a main topic.")
plt.tick_params('x', bottom=False, top=False, labelbottom=False)
plt.ylim([0,30000])
plt.show()

order_grouped = combined_dt.groupby(["main topic","sessionID"]).order.sum().value_counts().sort_index()
order_grouped/sum(order_grouped)
combined_dt.groupby(["main topic","sessionID"]).order.sum().value_counts().sort_index().plot.bar()
plt.title("Orders in a session for a main topic.")
plt.ylim([0,30000])
plt.xticks(rotation=45)
plt.show()

## Main topic and subtopic
item_dt[(item_dt['subtopics']=='[]')&item_dt['main topic']=='']


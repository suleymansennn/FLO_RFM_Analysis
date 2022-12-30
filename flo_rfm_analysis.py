"""
Business Problem:
Segmenting the customers of FLO, an online shoe store, wants to make sense according to these segments. It will be
designed accordingly and will be created according to this particular clustering. FLO, Wants to determine marketing
 strategies according to these segments.
"""

"""
Features
- master_id : Unique Customer Number
- order_channel : Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile)
- last_order_channel : The channel where the most recent purchase was made
- first_order_date : Date of the customer's first purchase
- last_order_channel : Customer's previous shopping history
- last_order_date_offline : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online : Total number of purchases made by the customer on the online platform
- order_num_total_ever_offline : Total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
- customer_value_total_ever_online :  Total fees paid for the customer's online purchases
- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import plotly.graph_objects as go
import plotly.io as pio

matplotlib.use("Qt5Agg")

colors = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF', "#CCA8E9", "#F67280"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df_ = pd.read_csv("rfm/flo_data_20k.csv")
df = df_.copy()
df.head()


def check_df(dataframe, head=5, tail=5):
    print("*" * 70)
    print(" Shape ".center(70, "*"))
    print("*" * 70)
    print(dataframe.shape)

    print("*" * 70)
    print(" Types ".center(70, "*"))
    print("*" * 70)
    print(dataframe.dtypes)

    print("*" * 70)
    print(" Head ".center(70, "*"))
    print("*" * 70)
    print(dataframe.head(head))

    print("*" * 70)
    print(" Tail ".center(70, "*"))
    print("*" * 70)
    print(dataframe.tail(tail))

    print("*" * 70)
    print(" NA ".center(70, "*"))
    print("*" * 70)
    print(dataframe.isnull().sum())

    print("*" * 70)
    print(" Quantiles ".center(70, "*"))
    print("*" * 70)
    print(dataframe.describe([.01, .05, .1, .5, .9, .95, .99]).T)

    print("*" * 70)
    print(" Uniques ".center(70, "*"))
    print("*" * 70)
    print(dataframe.nunique())


check_df(df)

df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

date_vars = df.columns[df.columns.str.contains("date")]
df[date_vars] = df[date_vars].apply(lambda x: pd.to_datetime(x))

df.info()
"""      
 0   master_id                          19945 non-null  object        
 1   order_channel                      19945 non-null  object        
 2   last_order_channel                 19945 non-null  object        
 3   first_order_date                   19945 non-null  datetime64[ns]
 4   last_order_date                    19945 non-null  datetime64[ns]
 5   last_order_date_online             19945 non-null  datetime64[ns]
 6   last_order_date_offline            19945 non-null  datetime64[ns]
 7   order_num_total_ever_online        19945 non-null  float64       
 8   order_num_total_ever_offline       19945 non-null  float64       
 9   customer_value_total_ever_offline  19945 non-null  float64       
 10  customer_value_total_ever_online   19945 non-null  float64       
 11  interested_in_categories_12        19945 non-null  object        
 12  new_total_expenditure              19945 non-null  float64       
 13  new_total_purchases                19945 non-null  float64       
"""

print(
    f'{df["new_total_purchases"].sum()} invoices were carried out from {df["first_order_date"].min()} to {df["last_order_date"].max()}')
print(
    f'{df["master_id"].nunique()} customer were served from {df["first_order_date"].min()} to {df["last_order_date"].max()}')
"""
100219.0 invoices were carried out from 2013-01-14 00:00:00 to 2021-05-30 00:00:00
19945 customer were served from 2013-01-14 00:00:00 to 2021-05-30 00:00:00
"""


def cat_plots(dataframe, cat_col):
    print("".center(100, "#"))
    print(dataframe[cat_col].value_counts())

    plt.figure(figsize=(15, 10))
    plt.suptitle(cat_col.capitalize(), size=16)
    plt.subplot(1, 2, 1)
    plt.title("Percentages")
    plt.pie(dataframe[cat_col].value_counts().values.tolist(),
            labels=dataframe[cat_col].value_counts().keys().tolist(),
            labeldistance=1.1,
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
            colors=colors,
            autopct='%1.0f%%')

    plt.subplot(1, 2, 2)
    plt.title("Countplot")
    sns.countplot(data=dataframe, x=cat_col, palette=colors)
    plt.tight_layout(pad=3)
    plt.show(block=True)


cat_plots(df, "order_channel")


def num_summary(dataframe, col_name, target):
    quantiles = [.01, .05, .1, .5, .9, .95, .99]
    print(dataframe.groupby(target)[col_name].describe(percentiles=quantiles))
    xlim = dataframe[col_name].describe(quantiles).T["99%"]

    plt.figure(figsize=(15, 10))
    plt.suptitle(col_name.capitalize(), size=16)
    plt.subplot(1, 3, 1)
    plt.title("Histogram")
    sns.histplot(dataframe[col_name], color="#FFB6B9")
    plt.xlim(0, xlim)

    plt.subplot(1, 3, 2)
    plt.title("Box Plot")
    sns.boxplot(data=dataframe, y=col_name, color="#F67280")
    plt.ylim(0, xlim)

    plt.subplot(1, 3, 3)
    sns.barplot(data=dataframe, x=col_name, y=target, palette=colors, estimator=np.mean)
    plt.title(f"Sum of {col_name.capitalize()} by {target.capitalize()}")
    plt.tight_layout(pad=1.5)
    plt.show(block=True)


for col in df.columns[df.columns.str.contains("new")]:
    num_summary(df, col, "order_channel")

df[["master_id", "new_total_expenditure", "new_total_purchases"]].sort_values("new_total_expenditure",
                                                                              ascending=False).head(10)
"""
                                  master_id  new_total_expenditure  new_total_purchases
11150  5d1c466a-9cfd-11e9-9897-000d3a38a36f             45905.1000             202.0000
4315   d5ef8058-a5c6-11e9-a2fc-000d3a38a36f             36818.2900              68.0000
7613   73fd19aa-9e37-11e9-9897-000d3a38a36f             33918.1000              82.0000
13880  7137a5c0-7aad-11ea-8f20-000d3a38a36f             31227.4100              11.0000
9055   47a642fe-975b-11eb-8c2a-000d3a38a36f             20706.3400               4.0000
7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f             18443.5700              70.0000
8068   d696c654-2633-11ea-8e1c-000d3a38a36f             16918.5700              70.0000
163    fef57ffa-aae6-11e9-a2fc-000d3a38a36f             12726.1000              37.0000
7223   cba59206-9dd1-11e9-9897-000d3a38a36f             12282.2400             131.0000
18767  fc0ce7a4-9d87-11e9-9897-000d3a38a36f             12103.1500              20.0000
"""
df[["master_id", "new_total_expenditure", "new_total_purchases"]].sort_values("new_total_purchases",
                                                                              ascending=False).head(10)
"""
                                  master_id  new_total_expenditure  new_total_purchases
11150  5d1c466a-9cfd-11e9-9897-000d3a38a36f             45905.1000             202.0000
7223   cba59206-9dd1-11e9-9897-000d3a38a36f             12282.2400             131.0000
8783   a57f4302-b1a8-11e9-89fa-000d3a38a36f             10383.4400             111.0000
2619   fdbe8304-a7ab-11e9-a2fc-000d3a38a36f              8572.2300              88.0000
6322   329968c6-a0e2-11e9-a2fc-000d3a38a36f              4240.3600              83.0000
7613   73fd19aa-9e37-11e9-9897-000d3a38a36f             33918.1000              82.0000
9347   44d032ee-a0d4-11e9-a2fc-000d3a38a36f              5184.0500              77.0000
10954  b27e241a-a901-11e9-a2fc-000d3a38a36f              5297.8800              75.0000
8068   d696c654-2633-11ea-8e1c-000d3a38a36f             16918.5700              70.0000
7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f             18443.5700              70.0000
"""

df["first_order_year"] = df["first_order_date"].dt.year
df["first_order_month"] = df["first_order_date"].dt.month_name()
df["first_order_day"] = df["first_order_date"].dt.day_name()

cat_plots(df, "first_order_year")


def preprocess(path):
    df_ = pd.read_csv(path)
    df = df_.copy()
    df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    date_vars = df.columns[df.columns.str.contains("date")]
    for col in date_vars:
        df[col] = pd.to_datetime(df[col])
    return df


df = preprocess("rfm/flo_data_20k.csv")


def rfm_table(dataframe):
    max_date = (dataframe["last_order_date"].max() + dt.timedelta(days=2))
    rfm = pd.DataFrame({
        "Recency": (max_date - dataframe["last_order_date"]),
        "Frequency": dataframe["new_total_purchases"],
        "Monetary": dataframe["new_total_expenditure"]
    })
    rfm["Recency"] = rfm["Recency"].apply(lambda x: x.days)
    return rfm


rfm = rfm_table(df)

rfm["Recency_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])
rfm["RF_Score"] = rfm["Recency_Score"].astype(str) + rfm["Frequency_Score"].astype(str)

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["Segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

rfm.groupby("Segment").agg({"Recency": ["mean", "count"],
                            "Frequency": ["mean", "count"],
                            "Monetary": ["mean", "count"]})
"""
                     Recency       Frequency        Monetary      
                        mean count      mean count      mean count
Segment                                                           
about_to_sleep      113.7851  1629    2.4015  1629  359.0090  1629
at_Risk             241.6068  3131    4.4724  3131  646.6102  3131
cant_loose          235.4442  1200   10.6983  1200 1474.4682  1200
champions            17.1066  1932    8.9343  1932 1406.6251  1932
hibernating         247.9495  3604    2.3940  3604  366.2671  3604
loyal_customers      82.5948  3361    8.3746  3361 1216.8186  3361
need_attention      113.8287   823    3.7278   823  562.1430   823
new_customers        17.9176   680    2.0000   680  339.9555   680
potential_loyalists  37.1559  2938    3.3043  2938  533.1845  2938
promising            58.9212   647    2.0000   647  335.6727   647
"""

sns.scatterplot(x="Frequency", y="Recency", data=rfm, hue="Segment")
plt.show(block=True)

check_df(rfm)

pio.renderers.default = "browser"

rfm_coordinates = {"champions": [3, 5, 0.8, 1],
                   "loyal_customers": [3, 5, 0.4, 0.8],
                   "cant_loose": [4, 5, 0, 0.4],
                   "at_Risk": [2, 4, 0, 0.4],
                   "hibernating": [0, 2, 0, 0.4],
                   "about_to_sleep": [0, 2, 0.4, 0.6],
                   "promising": [0, 1, 0.6, 0.8],
                   "new_customers": [0, 1, 0.8, 1],
                   "potential_loyalists": [1, 3, 0.6, 1],
                   "need_attention": [2, 3, 0.4, 0.6]}

palette = ["#282828", "#04621B", "#971194", "#F1480F", "#4C00FF",
           "#FF007B", "#9736FF", "#8992F3", "#B29800", "#80004C"]

fig = go.Figure()

df_shp = pd.DataFrame(rfm_coordinates).T.rename(
    columns={0: "y0", 1: "y1", 2: "x0", 3: "x1"}
)
df_shp["fillcolor"] = palette
df_shp.loc[:, ["x0", "x1"]] = df_shp.loc[:, ["x0", "x1"]] * 5

for segment, r in df_shp.iterrows():
    fig.add_shape(**r.to_dict(), opacity=0.6)
fig.update_layout(
    xaxis=dict(range=[0, 5], dtick=1, showgrid=False),
    yaxis=dict(range=[0, 5], showgrid=False),
    margin={"l": 0, "r": 0, "b": 0, "t": 0},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

df_txt = (
    rfm.groupby("Segment")
    .agg(avg_monetary=("Monetary", "mean"), number=("Monetary", "size"))
    .join(df_shp, how="right")
    .fillna(0)
)
fig.add_trace(
    go.Scatter(
        x=df_txt.loc[:, ["x0", "x1"]].mean(axis=1),
        y=df_txt.loc[:, ["y0", "y1"]].mean(axis=1),
        text=df_txt.index,
        customdata=df_txt.loc[:, ["avg_monetary", "number"]].astype(int).values,
        mode="text",
        texttemplate="<b>%{text}</b><br>Total Users:%{customdata[1]}<br>Average Monetary:%{customdata[0]}",
    )
)
fig.show()

rfm_cols = ["Recency", "Frequency", "Monetary"]

for col in rfm_cols:
    num_summary(rfm, col, "Segment")

# CASE 1: A new women's shoe brand will be included. The target audience (champions,loyal_customers) and women are
# determined as shoppers

case1 = df.loc[(rfm["Segment"] == "loyal_customers") | (rfm["Segment"] == "champions"),]

case1_index, case1_ids = zip(*[[row[0], row[1]] for row in case1.reset_index().to_numpy() if "KADIN" in row[-3]])
case1 = rfm.loc[case1_index,]
case1["master_id"] = case1_ids
case1 = case1[case1.columns.tolist()[-1:] + case1.columns.tolist()[:-1]]
# case1.to_csv("case1", index=False)

# CASE 2: A 40% discount on men's and children's products is planned. The target audience is (cant_loose,
# about_to_sleep, new_customers).

case2 = df.loc[
    (rfm["Segment"] == "about_to_sleep") | (rfm["Segment"] == "new_customers") | (rfm["Segment"] == "cant_loose"),]


case2_index, case2_ids = zip(
    *[[row[0], row[1]] for row in case2.reset_index().to_numpy() if ("COCUK" in row[-3] or "ERKEK" in row[-3])])

case2 = rfm.loc[case2_index,]
case2["master_id"] = case2_ids
case2 = case2[case2.columns.tolist()[-1:] + case2.columns.tolist()[:-1]]
# case2.to_csv("case2", index=False)

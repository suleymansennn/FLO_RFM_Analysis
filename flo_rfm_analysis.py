"""
İş Problemi
Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek
istiyor. Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar
oluşturulacak.

RFM analizi müşteri segmentasyonu için kullanılan bir tekniktir.
Müşterilerin satına alma alışkanlıkları üzerinden gruplara ayrılması ve bu gruplar özelinde stratejiler geliştirilebilmesini sağlar.
CRM çalışmaları için birçok başlıkta veriye dayalı aksiyon alma imkanı sağlar.

Peki bu gruplar nasıl oluşturulacak?
RFM analizi Recency, Frequency, Monetary değerlerinin skorlanması ile müşterileri sınıflara/segmentlere ayıran bir
tekniktir.

Recency: Müşteri en son ne zaman alışveriş yaptı?
Frequency: Müşterinin satın alma sıklığı/ toplam satın alma sayısı
Monetary: Müşterinin toplam harcadığı para

RFM Analizi bir şirketin en değerli müşterilerini, kaybetmekte olduğu müşterileri vb. gösterir.
"""

"""
Veri Seti Hikayesi:
Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) olarak 
yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

    master_id: Eşsiz müşteri numarası
    order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
    last_order_channel: En son alışverişin yapıldığı kanal
    first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
    last_order_date: Müşterinin yaptığı son alışveriş tarihi
    last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
    last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
    order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
    order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
    customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
    customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
    interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt

matplotlib.use("Qt5Agg")

colors = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF', "#CCA8E9", "#F67280"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df_ = pd.read_csv("crm/rfm/flo_data_20k.csv")
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

"""
    Veri Seti İle İlgili Aldığım Notlar:
        1. Tarih değişkenleri object olarak tanımlanmış. Tarih tipine değiştirilmesi gerekir.
        2. Veri setinde boş gözlem yok.
        3. Sipariş sayısında aykırı değerler var ancak bir skorlama işlemi yapılacağı için bir baskılama işlemine gerek yok.
        4. Online alışveriş sayısı daha fazla
        5. Ödenen toplam ücretlerde de bir aykırılık var. Ayrıca ortalama değer ve medyan arasındaki fark sağa doğru çarpıklık olduğunu gösteriyor.
        6. En son alışveriş yapılan kanal ile alışveriş kanalı arasında eşsiz değer farkı var?
            order_channel değişkeni sanırım müşterilerin üye olurken kullandığı kanal?
        7. Tüm müşteriler en az 1'er kere hem online hemde offline alışveriş yapmış
        8. Veri seti gruplanmış.  Bu yüzden yapılacak hesaplamalarda tekrardan bir gruplama yapılmasına gerek yok.
        9. İlgilenilen kategoriler değişkeni str olarak tanımlanmış?
"""

# Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_vars = df.columns[df.columns.str.contains("date")]
df[date_vars] = df[date_vars].apply(lambda x: pd.to_datetime(x))

# for col in date_vars:
  #  df[col] = pd.to_datetime(df[col])

df.info()

df.loc[1,"interested_in_categories_12"]
type(df.loc[1,"interested_in_categories_12"])
df["interested_in_categories_12"] = df["interested_in_categories_12"].apply(lambda x: x.replace("[", "").replace("]", "").split(","))

print(f'{df["new_total_purchases"].sum()} invoices were carried out from {df["first_order_date"].min()} to {df["last_order_date"].max()}')
print(f'{df["master_id"].nunique()} customer were served from {df["first_order_date"].min()} to {df["last_order_date"].max()}')

# Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

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

# Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
# Adım 7: En fazla sipariş veren ilk 10 müşteriyi sıralayınız.
df[["master_id", "new_total_expenditure", "new_total_purchases"]].sort_values("new_total_expenditure", ascending=False).head(10)
df[["master_id", "new_total_expenditure", "new_total_purchases"]].sort_values("new_total_purchases", ascending=False).head(10)




df["first_order_year"] = df["first_order_date"].dt.year
df["first_order_month"] = df["first_order_date"].dt.month_name()
df["first_order_day"] = df["first_order_date"].dt.day_name()

cat_plots(df, "first_order_year")
# 2019'da olan sıçrama ilgi çekici.

df.loc[df["first_order_year"]==2019, "first_order_month"].value_counts()

def preprocess(path):
    df_ = pd.read_csv(path)
    df = df_.copy()
    df["new_total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["new_total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    date_vars = df.columns[df.columns.str.contains("date")]
    for col in date_vars:
        df[col] = pd.to_datetime(df[col])
    return df

df = preprocess("crm/rfm/flo_data_20k.csv")

"""
Recency: Müşteri en son kaç gün önce alışveriş yaptı?
Frequency: Müşterinin satın alma sıklığı/ toplam satın alma sayısı
Monetary: Müşterinin toplam harcadığı para
"""

def rfm_table(dataframe):
    max_date = (dataframe["last_order_date"].max() + dt.timedelta(days=2))
    rfm = pd.DataFrame({
        "Recency": (max_date - dataframe["last_order_date"]),
        "Frequency": dataframe["new_total_num"],
        "Monetary": dataframe["new_total_value"]
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

"""
Segmentleri oluşturduk. Peki bunu neden yaptık?

    - Şirketimize en çok karı getiren müşteriler kimlerdir? Ürünlerimi en sık satın alan müşteriler kimlerdir?
    - Yeni müşteriler kimlerdir? 
    - Kaybetmekte olduğum veya yeni müşterileri nasıl şirketten daha fazla alışveriş yapması sağlanır?
    
"""

rfm.groupby("Segment").agg({"Recency": ["mean", "count"],
                           "Frequency": ["mean", "count"],
                           "Monetary": ["mean", "count"]})


sns.scatterplot(x="Frequency", y="Recency", data=rfm, hue="Segment")
plt.show(block=True)

check_df(rfm)


"""import plotly.graph_objects as go
import plotly.io as pio
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

palette = ["#282828", "#04621B", "#971194", "#F1480F",  "#4C00FF",
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
fig.show()"""

# Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm_cols = ["Recency", "Frequency", "Monetary"]

for col in rfm_cols:
    num_summary(rfm, col, "Segment")

# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel
# olarak iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden
# alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv
# dosyasına kaydediniz.

case1 = df.loc[(rfm["Segment"]=="loyal_customers") | (rfm["Segment"]=="champions"),]
# Segmenti loyal ve champions olan müşterilerin indexlerini tutarak, normal df'imizi bu indexlere göre filtreliyoruz.
# "KADIN" in df.loc[19943, "interested_in_categories_12"]

case1_index, case1_ids = zip( * [[row[0], row[1]] for row in case1.reset_index().to_numpy() if "KADIN" in row[-3]])
# case1_index[0:5] , case1_ids[0:5]
case1 = rfm.loc[case1_index, ]
case1["master_id"] = case1_ids
case1 = case1[case1.columns.tolist()[-1:] + case1.columns.tolist()[:-1]]
case1.to_csv("case1", index=False)

# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına
# kaydediniz.

case2 = df.loc[(rfm["Segment"]=="about_to_sleep") | (rfm["Segment"]=="new_customers") | (rfm["Segment"]=="cant_loose"),]
# "COCUK" in case2.loc[3, "interested_in_categories_12"] or "ERKEK" in case2.loc[3, "interested_in_categories_12"]

case2_index, case2_ids = zip( * [[row[0], row[1]] for row in case2.reset_index().to_numpy() if ("COCUK" in row[-3] or "ERKEK" in row[-3])])
case2 = rfm.loc[case2_index, ]
case2["master_id"] = case2_ids
case2 = case2[case2.columns.tolist()[-1:] + case2.columns.tolist()[:-1]]
case2.to_csv("case2", index=False)

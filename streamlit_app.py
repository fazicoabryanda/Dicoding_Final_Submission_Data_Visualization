import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Amazona_Dashboard",
                   page_icon="bar_chart:",
                   layout="wide")
st.logo("https://github.com/fazicoabryanda/Dicoding_Final_Submission_Data_Visualization/blob/44159968883544ba00acdcbee08c8f5d98868507/img/th.jpeg",size='large')
st.image("https://github.com/fazicoabryanda/Dicoding_Final_Submission_Data_Visualization/blob/44159968883544ba00acdcbee08c8f5d98868507/img/th.jpeg",width=200)
st.title("Amazona: Dashboard")


def produk_terlaris_unit(ecommerce):
    produk_terlaris_dalam_unit = ecommerce['product_categoy_name'].value_count()
    return produk_terlaris_dalam_unit

def produk_terlaris_harga(ecommerce):
    produk_terlaris_dalam_harga = ecommerce['price'].value_count()

def total_penjualan_semua_produk_unit(ecommerce):
    sum_product_category = ecommerce['product_category_name'].value_counts().sum()
    return sum_product_category

def total_penjualan_semua_produk_harga(ecommerce):
    sum_product_price = ecommerce['price'].sum()
    return sum_product_price

def rataan_penjualan(ecommerce):
    mean_product_price = ecommerce.groupby('product_category_name')['price'].mean()
    total_average_sales =  mean_product_price.sum()
    return total_average_sales


# Import Dataset
ecommerce = pd.read_csv("https://raw.githubusercontent.com/fazicoabryanda/Dicoding_Final_Submission_Data_Visualization/refs/heads/main/data/ecommerce_datacleaned.csv")
geolocation_df = pd.read_csv("https://raw.githubusercontent.com/fazicoabryanda/Dicoding_Final_Submission_Data_Visualization/refs/heads/main/data/geolocation_cleaned.csv")



# Membuat Sidebar
# Mula-mula ubah lakukan pembersihan data dengan mengurutkan 'order_purchase_timestamp', 'purchase_month_year'
datetime_columns = ['order_purchase_timestamp', 'purchase_month_year']
ecommerce.sort_values(by='order_purchase_timestamp', inplace=True)
ecommerce.reset_index(inplace=True)

for column in datetime_columns:
    ecommerce[column] = pd.to_datetime(ecommerce[column])

min_date = ecommerce['order_purchase_timestamp'].min()
max_date = ecommerce['order_purchase_timestamp'].max()

with st.sidebar:
    
    start_date, end_date = st.date_input(
        label='Waktu Pemesanan',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_ecommerce = ecommerce[(ecommerce['order_purchase_timestamp'] >= str(start_date)) & 
             (ecommerce['order_purchase_timestamp'] <= str(end_date))]

# Baris pertama
total_produk_unit = total_penjualan_semua_produk_unit(main_ecommerce)
total_produk_harga = total_penjualan_semua_produk_harga(main_ecommerce)
rataan_penjualan_keseluruhan = rataan_penjualan(main_ecommerce)


col1, col2, col3 =st.columns(3)
with col1:
    st.markdown('### Total Pemesanan:')
    st.subheader(total_produk_unit)

with col2:
    st.markdown('### Total Pendapatan:')
    st.subheader("{:.2f}".format(total_produk_harga))

with col3:
    st.markdown('### Rata-Rata Penjualan:')
    st.subheader("{:.2f}".format(rataan_penjualan_keseluruhan))

st.markdown("---")

# Baris kedua 
col1,col2 = st.columns(2)
with col1:
    st.markdown('#### 10 Besar Produk dengan Penjualan Terbanyak (Unit):')
    fig, ax = plt.subplots(figsize=(6,4))
    product_category_count = main_ecommerce['product_category_name'].value_counts()
    top_10_products = product_category_count[:10]
    sns.barplot(x=top_10_products.index, y=top_10_products.values)
    plt.xlabel('Nama Kategori Produk')
    plt.ylabel('Jumlah Penjualan')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("#### 10 Besar Produk dengan Pendapatan Penjualan Terbanyak:")
    top_10_products_by_revenue = main_ecommerce.groupby('product_category_name')['price'].sum().sort_values(ascending=False)[:10]

    # Membuat visualisasi data penjualan 10 produk terbanyak
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=top_10_products_by_revenue.index, y=top_10_products_by_revenue.values)
    plt.xlabel('Nama Kategori Produk')
    plt.ylabel('Total Pendapatan')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Baris ketiga
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 10 Besar Kota dengan Penjualan Terbanyak (Unit): \n")    
    customer_city_count = main_ecommerce['customer_city'].value_counts()
    top_10_cities = customer_city_count[:10]

    # Membuat plot bar
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_10_cities.index, y=top_10_cities.values, ax=ax)
    plt.xlabel('Nama Kota')
    plt.ylabel('Jumlah Penjualan')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Tampilkan grafik bar
    st.pyplot(fig)

with col2:
    st.markdown("#### 10 Besar Kota dengan Pendapatan Penjualan Terbanyak:")    
    top_10_cities_by_revenue = ecommerce.groupby('customer_city')['price'].sum().sort_values(ascending=False)[:10]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_10_cities_by_revenue.index, y=top_10_cities_by_revenue.values)
    plt.title('10 Kota dengan Pendapatan Terbanyak')
    plt.xlabel('Nama Kota')
    plt.ylabel('Total Pendapatan')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


st.markdown("---")

# Baris ketiga 
st.markdown("#### Tren Penjualan Produk :")
monthly_sales = main_ecommerce.groupby(main_ecommerce['order_purchase_timestamp'].dt.to_period('M'))['price'].sum()
# Plot the trend
fig, ax = plt.subplots(figsize=(12,6))
plt.plot(monthly_sales.index.to_timestamp(), monthly_sales.values, )
plt.xlabel('Bulan')
plt.ylabel('Total Pendapatan')
plt.grid(True)
st.pyplot(fig)

st.markdown("---")


# Kode untuk RFM
last_purchase_date = main_ecommerce['order_purchase_timestamp'].max()

# Kalkulasi dari rfm
rfm_df = main_ecommerce.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (last_purchase_date - x.max()).days,  # Recency
    'order_id': 'nunique',  # Frequency
    'price': 'sum'  # Monetary
})

rfm_df.rename(
    columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'MonetaryValue'
    },
    inplace=True
)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'MonetaryValue']])

kmeans = KMeans(n_clusters=5, random_state=42)  # 5 customer segments
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Membagikan segmen pelanggan ke beberapa cluster
def map_clusters(cluster):
    if cluster == 0:
        return 'Top Customers'
    elif cluster == 1:
        return 'Loyal Customers'
    elif cluster == 2:
        return 'Growing Loyalists'
    elif cluster == 3:
        return 'New Customers'
    else:
        return 'Emerging Customers'

rfm_df['Customer_Segment'] = rfm_df['Cluster'].apply(map_clusters)

rfm_df.drop(columns=['Cluster'], inplace=True)

st.markdown("#### Distribusi Segmen Pelanggan (RFM):")
customer_segment_count = rfm_df['Customer_Segment'].value_counts()

# Plot customer segments
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=customer_segment_count.index, y=customer_segment_count.values, ax=ax)
plt.xlabel('Customer Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# Visualizing RFM Distributions
st.markdown("### Distribusi Recency, Frequency, dan Monetary:")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Recency Distribution:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(rfm_df['Recency'], bins=30, kde=True, ax=ax)
    plt.xlabel('Recency (Days)')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("##### Frequency Distribution:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(rfm_df['Frequency'], bins=30, kde=True, ax=ax)
    plt.xlabel('Frequency (Number of Orders)')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    st.pyplot(fig)

with col3:
    st.markdown("##### Monetary Value Distribution:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(rfm_df['MonetaryValue'], bins=30, kde=True, ax=ax)
    plt.xlabel('Monetary Value (Total Purchase Amount)')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Display the RFM DataFrame
st.markdown("### RFM Analysis DataFrame:")
st.dataframe(rfm_df)

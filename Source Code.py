# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']='white'
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from datetime import date

# %%
from datetime import datetime

date_string = "21-08-2013"
format_string = "%d-%m-%Y"

try:
    parsed_date = datetime.strptime(date_string, format_string)
    print(parsed_date)
except ValueError as e:
    print(f"Error: {e}")

# %%
df = pd.read_csv("D:/DATA SCIENCE BOOTCAMP/Mini Project Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning/marketing_campaign_data.csv")
df.head()

# %%
df.info()

# %%
df.isna().sum()

# %%
df.describe ()

# %% [markdown]
# ## *Feature Extraction*

# %%
df2 = df.copy()

# %%
print(df2.columns)

# %%
# 1. Buat kolom 'Age' berdasarkan data 'Year_Birth'
df2['Age'] = date.today().year - df2['Year_Birth']

# 2. Feature Engineering: Conversion Rate
def safe_div(x, y):
    if y == 0:
        return 0
    return x / y

df2['Total_Purchases'] = df2['NumDealsPurchases'] + df2['NumWebPurchases'] + df2['NumCatalogPurchases'] + df2['NumStorePurchases']
df2['Conversion_Rate'] = df2.apply(lambda x: safe_div(x['Total_Purchases'], x['NumWebVisitsMonth']), axis=1)

# 3. Mengelompokkan umur ke beberapa kelompok
def age_grouping(age):
    if age >= 0 and age <= 1:
        return 'Infant'
    elif age >= 2 and age <= 4:
        return 'Toddler'
    elif age >= 5 and age <= 12:
        return 'Child'
    elif age >= 13 and age <= 19:
        return 'Teen'
    elif age >= 20 and age <= 39:
        return 'Adult'
    elif age >= 40 and age <= 59:
        return 'Middle Aged'
    else:
        return 'Senior Citizen'

df2['Age_Group'] = df2['Age'].apply(age_grouping)

# 4. Plot hubungan antara Conversion Rate dan Jenis User (Age Group)
plt.figure(figsize=(10, 6))
sns.barplot(x='Age_Group', y='Conversion_Rate', data=df2, order=['Infant', 'Toddler', 'Child', 'Teen', 'Adult', 'Middle Aged', 'Senior Citizen'])
plt.title('Conversion Rate per Age Group')
plt.xlabel('Age Group')
plt.ylabel('Conversion Rate')
plt.show()

# %%
df.duplicated().sum()

# %%
#change Dt_Customer dtype from object to datetime 64[ns]
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# %%
# Import the required library
from datetime import date
import pandas as pd

# ... (Assuming you have already imported the pandas library and loaded the dataframe as df2)

# Create total accepted campaign feature
df2['Total_Acc_Cmp'] = df2['AcceptedCmp1'] + df2['AcceptedCmp2'] + df2['AcceptedCmp3'] + df2['AcceptedCmp4'] + df2['AcceptedCmp5']

# Create total purchases feature
df2['Total_Purchases'] = df2['NumDealsPurchases'] + df2['NumWebPurchases'] + df2['NumCatalogPurchases'] + df2['NumStorePurchases']

# Create conversion rate feature
def safe_div(x, y):
    if y == 0:
        return 0
    return x / y

df2['cvr'] = df2.apply(lambda x: safe_div(x['Total_Purchases'], x['NumWebVisitsMonth']), axis=1)

# Create age feature
df2['Age'] = 2022 - df2['Year_Birth']

# Create age group
age_list = []
for i in df2['Age']:
    if 0 <= i <= 1:
        group = 'Infant'
    elif 2 <= i <= 4:
        group = 'Toddler'
    elif 5 <= i <= 12:
        group = 'Child'
    elif 13 <= i <= 19:
        group = 'Teen'
    elif 20 <= i <= 39:
        group = 'Adult'
    elif 40 <= i <= 59:
        group = 'Middle Aged'
    else:
        group = 'Senior Citizen'
    age_list.append(group)

df2['Age_Group'] = age_list

# Create total spend feature
df2['Total_Spent'] = df2['MntCoke'] + df2['MntFishProducts'] + df2['MntFruits'] + df2['MntMeatProducts'] + df2['MntSweetProducts'] + df2['MntGoldProds']

# Create amount of children feature
df2['NumChildren'] = df2['Kidhome'] + df2['Teenhome']

# Create total days joined
df2['Dt_Collected'] = date.today()
df2['Dt_Collected'] = df2['Dt_Collected'].astype('datetime64[ns]')
df2['Dt_Customer'] = pd.to_datetime(df2['Dt_Customer'])  # Convert 'Dt_Customer' column to datetime

df2['Dt_Days_Customer'] = df2['Dt_Collected'] - df2['Dt_Customer']
df2['Dt_Days_Customer'] = df2['Dt_Days_Customer'].dt.days

# %% [markdown]
# ## EDA

# %%
print(df2.columns)

# %%
df2.drop(['Unnamed: 0', 'ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Collected'], inplace=True, axis=1)

# %%
df2.head()

# %%
df3 = df2.copy()

# %%
df3[['Kidhome', 'Teenhome', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain', 'NumChildren', 'Total_Acc_Cmp']] = df3[['Kidhome', 'Teenhome', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain', 'NumChildren', 'Total_Acc_Cmp']].astype('object')


# %%
#looping to categorize cat include categorical columns and num include numeric columns
cat = []
num = []
dat = []
for col in df3.columns: 
    if df3[col].dtype == 'O':
        cat.append(col)
    elif df3[col].dtype == 'int64' or df3[col].dtype == 'float64':
        num.append(col)
    else:
        dat.append(col)

# %%
plt.figure(figsize= (15, 20))
for i in range(len(num)):
    plt.subplot(5, 4, i+1)
    sns.kdeplot(x = df3[num[i]])
    plt.tight_layout()

# %%
plt.figure(figsize= (10,15))
for i in range(len(num)):
    plt.subplot(5, 4, i+1)
    sns.boxplot(y = df3[num[i]], orient='v')
    plt.tight_layout()

# %%
plt.figure(figsize=(15, 25))
for i in range(len(cat)):
    plt.subplot(9, 2, i+1)
    ax = sns.countplot(y=cat[i], data=df3, palette='rocket', order=df3[cat[i]].value_counts().index)
    plt.bar_label(ax.containers[0])
    plt.tight_layout()

# %%
corr_matrix = df3.corr()

cmap = 'YlGnBu'

# Create the associations plot using seaborn's heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, cmap=cmap, annot=True, fmt=".2f")
plt.title('Associations Plot', fontsize=16)
plt.show()


# %%
df_task1 = df3.groupby('Age_Group').agg({'cvr':'sum'}).reset_index()
df_task1['sum_cvr'] = df_task1['cvr'].sum()
df_task1['pct'] = round((df_task1['cvr']/df_task1['sum_cvr'])*100, 2)
df_task1

# %%
custom_palette = sns.color_palette("Blues_r")

fig, ax = plt.subplots(figsize=(5, 5))

plt.title("Total of Customer Conversion Ratio\nBased on Age", fontsize=12, color='black', weight='bold', pad=15)
sns.barplot(x='Age_Group', y='cvr', data=df_task1, edgecolor='black', palette=custom_palette)  # Use the custom navy blue palette

plt.ylabel('Conversion Ratio')
plt.xlabel('Age Classification', labelpad=8)
plt.xticks(np.arange(3), ['Adult\n(25-44 years old)', 'Middle Aged\n(45-64 years old)', 'Senior Citizen\n(>64 years old)'])

plt.bar_label(ax.containers[0], padding=2)
plt.bar_label(ax.containers[0], ['14.19%', '48.05%', '37.76%'], label_type='center', color='white', weight='bold')  # Change label color to white

sns.despine()
plt.tight_layout()
plt.savefig('customer_cvr.png')


# %%
sns.set_palette("Blues_r")

plot = sns.jointplot(x=df3['Age'], y=df3['Income'], edgecolor='black')
plt.title('Age\nvs.\nIncome', fontsize=15, weight='bold')
plot.ax_marg_x.set_xlim(0, 80)
plot.ax_marg_y.set_ylim(0, 120000000)
plt.savefig('age_income_jointplot.png')
plt.show()


# %%
sns.set_palette("Blues_r")

# Create the joint plot for 'Age vs. Total Spent'
plot = sns.jointplot(x=df3['Age'], y=df3['Total_Spent'], edgecolor='black')
plt.title('Age\nvs.\nTotal Spent', fontsize=15, weight='bold')
plot.ax_marg_x.set_xlim(0, 80)
plt.savefig('age_totspent_jointplot.png')
plt.show()


# %%
sns.set_palette("Blues_r")

# Create the joint plot for 'Age vs. CVR'
plot = sns.jointplot(x=df3['Age'], y=df3['cvr'], edgecolor='black')
plt.title('Age\nvs.\nCVR', fontsize=15, weight='bold')
plot.ax_marg_x.set_xlim(0, 80)
plt.savefig('age_cvr_jointplot.png')
plt.show()


# %%
sns.set_palette("Blues_r")

# Create the joint plot for 'Income vs. CVR'
plot = sns.jointplot(x=df3['Income'], y=df3['cvr'], edgecolor='black')
plt.title('Income\nvs.\nCVR', fontsize=15, weight='bold')
plot.ax_marg_x.set_xlim(0, 120000000)
plt.savefig('income_cvr_jointplot.png')
plt.show()


# %%
sns.set_palette("Blues_r")

# Create the joint plot for 'Total Spent vs. CVR'
plot = sns.jointplot(x=df3['Total_Spent'], y=df3['cvr'], edgecolor='black')
plt.title('Total Spent\nvs.\nCVR', fontsize=15, weight='bold')
plt.savefig('totspent_cvr_jointplot.png')
plt.show()
plt.tight_layout()


# %% [markdown]
# ## Data Preprocessing

# %%
df_pre = df3.copy()

# %%
missing_value = df_pre.isna().sum()*100/len(df_pre)
print(round(missing_value, 4).sort_values(ascending=False))

# %%
#fill income missing value wih median
df_pre['Income'] = df_pre['Income'].fillna(df_pre['Income'].median())

# %%
df_pre.duplicated().sum()

# %%
df2[df2.duplicated(keep='first')].head(10)

# %%
df2[df2.duplicated(keep='last')].head(10)

# %% [markdown]
# Dari fungsi .duplicated() yang sudah dijalankan diatas, terdapat 183 baris data yang duplikat. Namun, ketika dilihat secara manual, ternyata tidak ada duplikat pada baris manapun. Untuk itu, dibuatlah keputusan untuk tidak menghilangkan data dari datu baris manapun.

# %% [markdown]
# ## *Drop Unnecesary Feature*

# %%
df_pre.drop(['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',
                'NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases',
                'MntCoke','MntFishProducts','MntFruits','MntMeatProducts','MntSweetProducts', 'MntGoldProds',
                'Kidhome','Teenhome', 'Response'], inplace=True, axis=1)

# %%
pip install dython

# %%
df_pre.info()

# %%
df_pre.shape

# %% [markdown]
# ### Feature Selection

# %%
corr_matrix = df_pre.corr()

cmap = 'YlGnBu'

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, cmap=cmap, annot=True, fmt=".2f")
plt.title('Associations Plot', fontsize=16)
plt.show()

plt.savefig('associations_plot.png')
plt.show()


# %% [markdown]
# Kolom yang dipilih berdasarkan RFM degan metode reduce dimensionality diantaranya:
# 
# R: Recency
# F: Total_Purchases
# M: Spent
# L: Age
# C: Total_Acc_Cmp

# %%
df_m = df_pre.copy()
df_m = df_m[['Recency', 'Total_Purchases', 'Total_Spent', 'Dt_Days_Customer', 'Age']]
df_m.columns = ['R', 'F', 'M', 'L', 'C']
df_m.describe(include='all')

# %%
corr_matrix = df_m.corr()

cmap = 'YlGnBu'

plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, cmap=cmap, annot=True, fmt=".2f")
plt.title('Associations Plot', fontsize=16)
plt.show()


# %%
cols = df_m.columns

plt.figure(figsize= (15, 20))
for i in range(len(cols)):
    plt.subplot(6, 2, i+1)
    sns.kdeplot(x = df_m[cols[i]])
    plt.tight_layout()

# %%
cols = df_m.columns
plt.figure(figsize= (10,15))
for i in range(len(cols)):
    plt.subplot(4, 4, i+1)
    sns.boxplot(y = df_m[cols[i]], orient='v')
    plt.tight_layout()

# %%
for col in cols:
    high_cut = df_m[col].quantile(q=0.99)
    low_cut= df_m[col].quantile(q=0.01)
    df_m.loc[df_m[col]>high_cut,col]=high_cut
    df_m.loc[df_m[col]<low_cut,col]=low_cut

# %%
cols = df_m.columns
plt.figure(figsize= (10,15))
for i in range(len(cols)):
    plt.subplot(4, 4, i+1)
    sns.boxplot(y = df_m[cols[i]], orient='v')
    plt.tight_layout()

# %% [markdown]
# ### Feature Transformation

# %%
#column M distribution is right-skewed

plt.figure(figsize= (5, 5))
sns.kdeplot(x = df_m['M'])
plt.tight_layout()

# %%
#log tranformation on column `M`

df_m_log = df_m.copy()
df_m_log['M'] = np.log(df_m['M'])

plt.figure(figsize= (5, 5))
sns.kdeplot(x = df_m_log['M'])
plt.tight_layout()

# %%
df_m_log.describe()

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %%
col_name = list(df_m_log.columns)

mm = MinMaxScaler()
df_std_mm = mm.fit_transform(df_m_log)
df_std_mm = pd.DataFrame(df_std_mm, columns=col_name)
df_std_mm.sample(10)

# %%
# Custom colors for each variable
variable_colors = ['#4c72b0', '#8172b3', '#937860', '#ccb974', '#b0495e']

plt.figure(figsize=(10, 5))
plt.title('Distribution MinMax Scaler')

sns.kdeplot(df_std_mm['R'], label='Recency', color=variable_colors[0])
sns.kdeplot(df_std_mm['F'], label='Total_Purchases', color=variable_colors[1])
sns.kdeplot(df_std_mm['M'], label='Spent', color=variable_colors[2])
sns.kdeplot(df_std_mm['L'], label='Dt_Days_Customer', color=variable_colors[3])
sns.kdeplot(df_std_mm['C'], label='Age', color=variable_colors[4])

plt.xlabel(None)
plt.legend()
plt.show()


# %%
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_std_mm.describe()

# %% [markdown]
# ## Model and Evaluation

# %% [markdown]
# ### Inertia

# %%
inertia = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df_std_mm)
    inertia.append(kmeans.inertia_)

line_color = '#FF5733'
scatter_color = '#007FFF'

plt.figure(figsize=(10, 5))
plt.title('Inertia Evaluation Score', fontsize=16, weight='bold')
sns.lineplot(x=range(2, 11), y=inertia, color=line_color, linewidth=4)
sns.scatterplot(x=range(2, 11), y=inertia, s=300, color=scatter_color, linestyle='--')
plt.xlabel('n_clusters', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.show()


# %% [markdown]
# ### Silhoutte

# %%
range_n_clusters = list(range(2, 11))
arr_silhouette_score_euclidean = []

for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i).fit(df_std_mm)
    preds = kmeans.predict(df_std_mm)
    score_euclidean = silhouette_score(df_std_mm, preds, metric='euclidean')
    arr_silhouette_score_euclidean.append(score_euclidean)

# Set different colors for the line plot and scatter plot
line_color = '#FF5733'
scatter_color = '#007FFF'

fig, ax = plt.subplots(figsize=(10, 5))
plt.title('Silhouette Evaluation Score', fontsize=16, weight='bold')
sns.lineplot(x=range(2, 11), y=arr_silhouette_score_euclidean, color=line_color, linewidth=4)
sns.scatterplot(x=range(2, 11), y=arr_silhouette_score_euclidean, s=300, color=scatter_color, linestyle='--')
plt.xlabel('n_clusters', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.show()


# %%
df_std_cluster = df_std_mm.copy()
df_cluster = df_m.copy()

kmeans = KMeans(n_clusters=4, random_state=0).fit(df_std_mm)
df_std_cluster['clusters'] = kmeans.labels_
df_cluster['clusters'] = kmeans.labels_

# %% [markdown]
# ### PCA

# %%
from sklearn.decomposition import PCA 

# %%
pca = PCA(n_components=2)

pca.fit(df_std_mm)
pcs = pca.transform(df_std_mm)

df_pca = pd.DataFrame(data = pcs, columns = ['PC 1', 'PC 2'])
df_pca['clusters'] = df_cluster['clusters']
df_pca.sample(10)

# %%
fig, ax = plt.subplots(figsize=(10,8))
plt.title("2-D Visualization of Customer Clusters\nWih PCA", fontsize=15, weight='bold')
sns.scatterplot(
    x="PC 1", y="PC 2",
    hue="clusters",
    edgecolor='black',
    #linestyle='--',
    data=df_pca,
    palette=['blue','orange','green','red'],
    s=160,
    ax=ax
);

# %%
fig, ax = plt.subplots(figsize=(10,8))
plt.title("2-D Visualization of Customer Clusters\nWih PCA", fontsize=15, weight='bold')
sns.scatterplot(
    x="PC 1", y="PC 2",
    hue="clusters",
    edgecolor='black',
    #linestyle='--',
    data=df_pca,
    palette=['#4c72b0', '#8172b3', '#937860', '#ccb974'],
    s=160,
    ax=ax
);

# %%
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
plt.title("3-D Visualization of Customer Clusters\nBased on its Characteristics", fontsize=15, weight='bold')
cluster_palette = ['#4c72b0', '#8172b3', '#937860', '#ccb974']
ax.scatter(df_cluster['R'][df_cluster.clusters == 0], df_cluster['F'][df_cluster.clusters == 0], df_cluster['M'][df_cluster.clusters == 0], c=cluster_palette[0], s=100, edgecolor='black', label='High-Valued Customer')
ax.scatter(df_cluster['R'][df_cluster.clusters == 1], df_cluster['F'][df_cluster.clusters == 1], df_cluster['M'][df_cluster.clusters == 1], c=cluster_palette[1], s=100, edgecolor='black', label='Low-Valued Customer')
ax.scatter(df_cluster['R'][df_cluster.clusters == 2], df_cluster['F'][df_cluster.clusters == 2], df_cluster['M'][df_cluster.clusters == 2], c=cluster_palette[2], s=100, edgecolor='black', label='High-Valued Frequent Customer')
ax.scatter(df_cluster['R'][df_cluster.clusters == 3], df_cluster['F'][df_cluster.clusters == 3], df_cluster['M'][df_cluster.clusters == 3], c=cluster_palette[3], s=100, edgecolor='black', label='Low-Valued Frequent Customer')

plt.xlabel('Recency')
plt.ylabel('Total Purchases')
ax.set_zlabel('Total Spent')
plt.legend(title='Cluster:')
plt.show()


# %% [markdown]
# Insight

# %%
display(df_cluster.groupby('clusters').agg(['mean','median', 'max', 'min']))

# %%
df_totalc = df_cluster.groupby('clusters').agg({'R':'count'}).reset_index()
df_totalc = df_totalc.rename(columns={'R':'total_customers'})
df_totalc['sum_customers'] = df_totalc['total_customers'].sum()
df_totalc['pct'] = round((df_totalc['total_customers']/df_totalc['sum_customers'])*100, 2)
df_totalc

# %%
cluster_colors = ['#4c72b0', '#8172b3', '#937860', '#ccb974']

fig, ax = plt.subplots(figsize=(6, 5))
plt.title("Total of Customers Each Cluster", fontsize=15, color='black', weight='bold', pad=15)

sns.barplot(x='clusters', y='total_customers', data=df_totalc, edgecolor='black', palette=cluster_colors)

plt.xlabel('Clusters', fontsize=11)
plt.ylabel('Total Customer', fontsize=11)
plt.bar_label(ax.containers[0], padding=2)
plt.bar_label(ax.containers[0], ['28.93%', '22.01%', '27.28%', '21.79%'], label_type='center', color='white', weight='bold')

sns.despine()
plt.tight_layout()
plt.show()


# %%
# Palet warna untuk setiap klaster
cluster_colors = ['#4c72b0', '#8172b3', '#937860', '#ccb974']

# Ubah nama kolom di df_cls
df_cls = df_cluster.copy()
df_cls.rename(columns={'R':'Recency','F':'Total Purchases','M':'Total Spent','L':'Membership Days','C':'Age'}, inplace=True)

cls = df_cls.columns.drop('clusters')

plt.figure(figsize= (15, 8))
for i in range(len(cls)):
    plt.subplot(2, 3, i+1)
    sns.boxenplot(x=df_cluster['clusters'], y=df_cls[cls[i]], palette=cluster_colors)
    plt.tight_layout()

plt.show()


# %%
df_pre['Clusters'] = kmeans.labels_
df_pre.head()

# %%
df_age_clus = df_pre.groupby(['Clusters', 'Age_Group']).agg({'Education':'count'}).reset_index()
df_age_clus = df_age_clus.rename(columns={'Education':'total_customers'})
df_age_clus['sum_customers'] = df_age_clus['total_customers'].sum()
df_age_clus['pct'] = round((df_age_clus['total_customers']/df_age_clus['sum_customers'])*100, 2)
df_age_clus

# %%
# Custom color palette
custom_palette = ['#4c72b0', '#8172b3', '#937860']

fig, ax = plt.subplots(figsize=(9, 5))
plt.title("Total of Customers Each Cluster\nBased on Age", fontsize=15, color='black', weight='bold', pad=30)

sns.barplot(x='Clusters', y='total_customers', data=df_age_clus, hue='Age_Group', edgecolor='black', palette=custom_palette)

plt.text(x=-0.8, y=370, s="Middle Aged Customer dominated on each cluster (>13% of total customer).", fontsize=12, fontstyle='italic')
plt.xlabel('Clusters', fontsize=11)
plt.xticks(np.arange(4), ['High-Valued Customer', 'Low-Valued Customer', 'High-Valued Frequent Customer', 'Low-Valued Frequent Customer'], rotation=5)
plt.ylabel('Total Customer', fontsize=11)
plt.ylim(0, 350)
plt.legend(prop={'size':8}, loc='best')

# Update the bar_label text colors
plt.bar_label(ax.containers[0], padding=2,)
plt.bar_label(ax.containers[1], padding=2,)
plt.bar_label(ax.containers[2], padding=2,)
plt.bar_label(ax.containers[0], ['2.9%', '3.17%', '2.63%', '4.73%'], label_type='center', color='white', weight='bold', fontsize=8)
plt.bar_label(ax.containers[1], ['14.02%', '13.08%', '14.87%', '13.39%'], label_type='center', color='white', weight='bold', fontsize=8)
plt.bar_label(ax.containers[2], ['12.01%', '5.76%', '9.78%', '3.66%'], label_type='center', color='white', weight='bold', fontsize=8)

sns.despine()
plt.tight_layout()
plt.show()


# %%
plot = sns.jointplot(x=df_pre['Income'], y=df_pre['cvr'], hue=df_pre['Clusters'], edgecolor='black', palette=['#4c72b0', '#8172b3', '#937860', '#ccb974'])
plt.title('Income\nvs.\nCVR', fontsize=15, weight='bold')
plot.ax_marg_x.set_xlim(0, 120000000)
plt.show()

# %%
display(df_pre.groupby('Clusters').describe(include='all'))

# %% [markdown]
# ## *Interpreation Customer Summary*

# %% [markdown]
# Visualization results using PCA with 2 main PCs show that the customer clusters are perfectly separated. The K-Means Clustering Algorithm using the RFMLC Method produces 4 clear customer clusters in this dataset.

# %% [markdown]
# 1. High-Valued Customer (Cluster 0):
# 
# Cluster 0 has 648 customers (28.93% of the total subscribers). They have high novelty (73 days on average) and high total purchases (21 items on average), indicating high spending on our platform (about 1 million per year). The majority of customers in this group are middle-aged customers (45-64 years) of 48.46%, most have 1 child, and have the highest average income (around IDR 65 million per year) with low web visits per month (average -average 4 times).
# 
# 2. Low-Valued Customer (Cluster 1):
# 
# • 493 customers (22.01% of the total) in this group.
# • Highest average novelty (74 days) and low purchases (8 items on average), meaning they spend less and less on our platform (around 92k per year).
# • Domination by 59.43% middle aged customers (45-64 years) with 1 child and average income (around 36 million per year) and high monthly web visits (6 times on average).
# 
# 3. High-Valued Frequent Customers (Cluster 2):
# 
# • 611 customers (27.28% of the total) in this group.
# • Low average novelty (23 days) and high purchases (21 items on average), meaning they shop frequently and a lot on our platform (around 989k per year).
# • Domination by 54.5% middle aged customers (45-64 years) with 1 child and average income (about 65 million per year) with low monthly web visits (4 times average).
# 
# 4. Low-Valued Frequent Customers (Cluster 3):
# 
# • 488 customers (21.79% of the total) in this group.
# • High average recency (24 days) and lowest purchases (average 7 items), meaning they spend often but little on our platform (around 75 thousand per year).
# • Domination by 61.48% middle aged customers (45-64 years) with 1 child and average income (around 35 million per year) with high monthly web visits (6 times on average).
# 
# 
# 
# 
# 

# %% [markdown]
# ## *Recommendation*

# %% [markdown]
# Insights:
# 
# Create a membership tier program (Platinum, Gold, Silver, Bronze) with different privileges for each customer group (High Rated Customer, High Rated Frequent Customer, Low Rated Frequent Customer, Low Rated Customer).
# 
# Prioritize focusing on a group of High-Valued Customers to prevent churn. Improve service, after-sales maintenance and product quality. Provide Platinum membership with discounts, promotions and free shipping to encourage more frequent shopping.
# 

# %% [markdown]
# 



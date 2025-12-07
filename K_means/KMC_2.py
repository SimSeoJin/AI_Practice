import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 아 이게 고객별 카드 내역이 아니라 실시간으로 사람들이 사용한 결제내역이네 ( cc_num이 중복 될 수 있네 )
file_url = r'C:\Users\SeoJin\OneDrive\Desktop\폴더\부경대_수업\3-2\기계학습1\Practice\Data\customer.csv'
data=pd.read_csv(file_url)
# print(data.head())

# 전처리
# 범주형 특징을 수치형으로 변환  ( 범주형 데이터를 수치형 데이터로 변환해서 저장한 변수를 더미 변수라고 한다. )
data_dummy = pd.get_dummies(data,columns=['category'])

# print(data_dummy)
# 2열부터 시작하는 각 특징 이름 리스트 생성 ( 2열부터 범주형데이터의 수치형으로 바꾼 컬럼들이 생성되어 있음. )
cat_list = data_dummy.columns[2:]
for i in cat_list:
    data_dummy[i] = data_dummy[i] * data_dummy['amt']

# print(data_dummy.head()) # 각 고객들이 주로 쓴 경비에 대해서만 값이 있고 나머지는 다 0
# customer_agg = data_dummy.grouby('cc_cum')
# grouby를 하게 되면 DataFrameGroupby객체가 반환돼서 주소값이 반환된다. 
customer_agg = data_dummy.groupby('cc_num').sum()
scaler = StandardScaler() # 표준화 스케일링
customer_agg_scaled = scaler.fit_transform(customer_agg) # 넘파이로 반환.

# 🚨 문제 해결: NumPy 배열을 다시 Pandas 데이터프레임으로 변환
# 원래의 컬럼 이름과 인덱스(cc_num)를 복원합니다.
customer_agg_scaled_df = pd.DataFrame(
    customer_agg_scaled, 
    columns=customer_agg.columns,
    index=customer_agg.index
)

k_model=KMeans(n_clusters=4)
k_model.fit(customer_agg_scaled_df)

labels = k_model.predict(customer_agg_scaled_df)
# 스케일링 하고 나서 넘파이가 반환되는데 그 상태에서 labels컬럼을 추가하면 에러발생함.
# 즉, 데이터 프레임으로 변환해주는 우의 과정이 필요함
customer_agg_scaled_df['label'] = labels 
scaled_df_mean = customer_agg_scaled_df.groupby('label').mean()
scaled_df_count = customer_agg_scaled_df.groupby('label').count()['category_travel']
scaled_df_count = scaled_df_count.rename('count') # 이름 변경
scaled_df_all = scaled_df_mean.join(scaled_df_count) # 데이터 합치기

# 실루엣 계수 확인
silhouette = []

for k in range(2,10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(customer_agg_scaled_df)
    labels = k_model.predict(customer_agg_scaled_df)
    silhouette.append(silhouette_score(customer_agg_scaled_df,labels))

sns.lineplot(x=range(2,10),y=silhouette)
plt.show()






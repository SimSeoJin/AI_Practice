import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 

# 모델 학습 연습
file_url = r'C:\Users\SeoJin\OneDrive\Desktop\폴더\부경대_수업\3-2\기계학습1\Practice\Data\example_cluster.csv'
data = pd.read_csv(file_url)

plt.figure(figsize=(10,10))
sns.scatterplot(x='var_1',y='var_2',data=data)
# plt.show() => 산점도를 보면 데이터들이 3개의 군집화로 되어 있는 걸 확인
# => 3개의 군집을 만들어주면 되겠다
# 클러스터 3개 = 중심점( centroid 3개 )
KMeans_model = KMeans(n_clusters=3,random_state=100)

KMeans_model.fit(data)
data['label'] = KMeans_model.predict(data)  # 예측값을 label란 특징을 만들어서 저장

# hue는 범주형 변수를 지정해서 각 값에 따라 색깔을 다르게 구분.( rainbow색으로)
sns.scatterplot(x='var_1',y='var_2',data=data,hue='label',palette='rainbow')


# 모델 군집 응집도 및 분리도 측정 => 이너셔(inertia)/응집도, 실루엣(Shilhouette)/응집도 및 분리도
# 클러스터의 수의 따라 응집도 증가 실험
plt.figure(figsize=(10,10))
distance = []
for k in range(2,10):
    KMeans_model_2 = KMeans(n_clusters=k,random_state=100) # k-means 모델 생성
    KMeans_model_2.fit(data)
    distance.append(KMeans_model_2.inertia_) # 이너셔를 거리 리스트에 저장 

sns.lineplot(x=range(2,10),y=distance)
plt.show()
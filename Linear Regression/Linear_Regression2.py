import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

file_url = r"C:\Users\SeoJin\OneDrive\Desktop\폴더\부경대_수업\3-2\기계학습1\Practice\Linear Regression\Linear Regression2.csv"
data = pd.read_csv(file_url)

# 전처리: train set이랑 test set 나누기
X = data[['age','sex','bmi','children','smoker']]
y = data['charges']
# 훈련 데이터(입력,출력)과 테스트 세트 자동으로 몇프로 분할 (0.2 => 20% )
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)
print(X_train)

model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)

comparsion = pd.DataFrame({'actual':y_test,'pred':pred})
print(comparsion)

plt.figure(figsize=(10,10))
sns.scatterplot(x='actual',y='pred',data=comparsion)
# plt.show()

# 성능 평가
print(mean_squared_error(y_test,pred)) # MSE
# print(mean_squared_error(y_test,pred,squared=False)) # RMSE 이거 함수 없어졌나봄
print(model.score(X_train,y_train)) # 결졍계수(R^2) 1에 가까울 수록 좋음

# 학습된 모델의 가중치
print(model.coef_)
print(pd.Series(model.coef_,index=X.columns))

# bias
print(model.intercept_)
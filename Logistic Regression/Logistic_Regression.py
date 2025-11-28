import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 타이타닉 데이터
file_url = r"C:\Users\SeoJin\OneDrive\Desktop\폴더\부경대_수업\3-2\기계학습1\Practice\Logistic Regression\Logistic Regression.csv"
data = pd.read_csv(file_url)

# 맨앞 5행
print(data.head())

# 그래서 뭐 할려고 저 데이터로 ? => 생존 여부 예측 ( 근데 나이나 성 이런거 가지고 생존 여부를 판단할 수 있나.. ?)

# 데이터 프레임에서 숫자 열만 선택
data_numeric = data.select_dtypes(include=([float,int]))

# 코를레이션 확인 ( 상관관계 이런 거 보여주기도 하네 "음/양 상관관계인지")
print(data_numeric.corr())

sns.heatmap(data_numeric.corr())

# 전처리 ( 카테고리 변수 변환하기 (더미 변수와 원-핫 인코딩))
# data 데이터 프레임에서 name란 열에 고유값의 수를 계산하는 메서드
print(data['Name'].nunique()) # number unique약자구나
data = data.drop(['Name','Ticket'], axis=1)

'''
# 원핫 인코딩 수행하는 함수 ( 범수형 데이터를 숫자 형태로 변환, Sex와Embared 열은 삭제 되고 원핫 인코딩 된 새로운 데이터 프레임이 반환된다. )
# 예로들어서 Sex라는 카테고리에서 나올 수 있는 범주 Female,male이니깐 이를 두개의 열로 만듬.
data=pd.get_dummies(data, columns=['Sex','Embarked']) 
'''
# 다중공선성을 밪이하기 위함 ( 그러니깐 Sex의 경우 범주가 두개라서 두 개의 범주에 대해 k개의 더미 변수를 만들면
# k번째 더미 변수는 나머지 더미 변수를 통해 완벽하게 예측가능해지니깐 자세한 내용은 필요시 찾아보자..)
data = pd.get_dummies(data,columns=['Sex','Embarked'],drop_first=True) # k-1번째 더미 변수 삭제

X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)

# 수집한 데이터의 전처리 과정 후 모델 학습
model = LogisticRegression()
model.fit(X_train,y_train)

# 학습한 모델로 테스트
pred=model.predict(X_test)

# 이건 생략해도 됌
pred_result=np.array(model.predict(X_test)).reshape(-1,1)
print(pred_result)

# 예측 모델 평가하기
print(accuracy_score(y_test,pred))

# 모델 가중치 확인 ( 모델이 학습되고 난 후 각 특징 가중치 )
print(pd.Series(model.coef_.flatten(),index=X.columns))

'''
# Feature Enginnering
# 위에 로지스틱 모델을 보면 정확도가 0.7이라 성능이 그렇게 좋지 못함
# 그래서 성능을 좀 더 올리기 위해 엔지니어링 작업을 수행
data['family'] = data['Sibsp'] + data['Parch']
data.drop(['Sibsp','Parch'],axis=1,inplace=True)
print(data.head())

X = data.drop(['Survived'],axis=1) # Input
y = data['Survived'] # Target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)
model = LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
accuracy_score(y_train,pred)
'''
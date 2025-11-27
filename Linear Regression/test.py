import numpy as np
X = 2 * np.random.rand(100, 1) # 0~2사이의 값을 갖는 100개의 행과 1개의 열
y = 4 + 3 * X + np.random.randn(100, 1) # f(x) = 3x+4+b // y=f(x)는 타겟값 특징이 한개라서 가중치는 3으로 둿나보다 np.random.randn(100,1)은 노이즈 

X_b = np.c_[X, np.ones((100, 1))]  # 모든 샘플에 bias term 추가 np.c_함수 => 열방향으로 합쳐주는 넘파이 함수

print(X)
# HW 01
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)  # X는 100*1의 행렬이며 그 안에 0~1사이의 랜덤 변수를 갖는다.
y = 4 + 3 * X + np.random.randn(100, 1)  # Y는 4에 X에 3을 곱한것을 더한 후 100*1행렬크기에 0~1사이의 난수를 더한다.

# plt.plot(X, y, "b.")  # X,y 의 좌표를 갖는 점을 찍는다
# plt.xlabel("$x_1$", fontsize=18)  # x축의 이름은 x1이라 하고 폰트 크기는 18이다.
# plt.ylabel("$y$", rotation=0, fontsize=18)  # y축의 이름은 y이고 글씨를 0도 회전 시켜라
# plt.axis([0, 2, 0, 15])  # x축은 0~2까지의 범위이고 Y축은 0~15까지의 범위이다.
# plt.show()  # 그래프를 그려라

# 100개의 랜덤 좌표를 찍고 그 좌표는 대체로 y=3x+4에 근사한다.


X_b = np.c_[np.ones((100, 1)), X]  # 1로 초기화된 100 * 1크기의 행렬을 만들어 그 행렬을 X와 열의 방향으로 붙인다.
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(
    y)  # X_b를 transpose시켜 X_b와 곱한것의 역행렬에 X_b의 transpose를 곱한후 다시 y를 곱한다.
# 위 식이 나온 이유는 원래라면 x끼리 내적을 한후 역행렬을 구한것에 다시 x와 y를 내적시키고 싶지만 역행렬을 구하려면 정사각 행렬이야 하므로 어쩔 수 없이 1로 초기화된 100 * 1 행렬을 붙여준 것이다.


X_new = np.array([[0], [2]])  # 행렬을 만든다
X_new_b = np.c_[np.ones((2, 1)), X_new]  # X_new와 1로 초기화된 열벡터를 서로 합친다.
y_predict = X_new_b.dot(theta_best)  # X_new_b 와 theta_best를 곱한다.

# plt.plot(X_new, y_predict, "r-", linewidth=2, label="prediction")
# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.legend(loc="upper left", fontsize=14)
# plt.axis([0, 2, 0, 15])
# plt.show()

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()  # 모델은 linear regression을 사용하겠다.
lin_reg.fit(X, y)  # 모델에 X, y 값들을 적용하겠다.

# print(lin_reg.intercept_)           # bias 의 예측값을 얻어냄
# print(lin_reg.coef_)                # 기울기 w의 예측값을 얻어냄

# print(lin_reg.predict(X_new))        # 6번

# theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond = 1e-6)

# print (theta_best_svd)                  # 7번
# print(np.linalg.pinv(X_b).dot(y))           #8번 [[4.08409618]
                                            #   [2.96693676]]

### 경사 하강법을 이용한 선형회귀 접근 ###
eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2, 1)       # 우선 theta를 임의의 값으로 초기화
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)   # 기울기를 구함
    theta = theta - eta*gradients               # 기울기에 learning rate를 곱한값을 빼주어 theta를 조정해줌

print(theta)
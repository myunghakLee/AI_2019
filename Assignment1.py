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
theta = np.random.randn(2, 1)  # 우선 theta를 임의의 값으로 초기화
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # 기울기를 구함
    theta = theta - eta * gradients  # 기울기에 learning rate를 곱한값을 빼주어 theta를 조정해줌

theta_path_bgd = []


def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=18)


np.random.seed(42)
theta = np.random.randn(2, 1)
# plt.figure(figsize=(10, 4))
# plt.subplot(131);
# plot_gradient_descent(theta, eta=0.02)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(132);
# plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
# plt.subplot(133);
# plot_gradient_descent(theta, eta=0.5)
# plt.show()

#
### 스토캐스틱 경사 하강법을 사용한 선형 회귀 접근 ###
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)
n_epochs = 50  # 총 50세대를 거칠 것이다.
t0, t1 = 5, 50  # 매 반복마다 학습률을 정하여 learning rate를 다르게 줄 것이다.


def learning_schdule(t):  # 학습률은 다음과 같이 초기화 할 것이다.
    return t0 / (t + t1)


theta = np.random.randn(2, 1)  # 처음에는 theta값을 random하게 초기화 할 것이다.

# for epoch in range(n_epochs):
#     for i in range(m):
#         if epoch == 0 and i < 20:  # 첫번째 세대의 20번batch까지만 그래프를 그리겠다
#             y_predict = X_new_b.dot(theta)
#             style = "b-" if i > 0 else "r--"  # i가 0초과일때는 파란 실선 아닐 때는 빨간 점선으로 그린다.
#             plt.plot(X_new, y_predict, style)
#         random_index = np.random.randint(m)  # 순서대로 가 아닌 아무점이나 random으로 선택하여 theta값을 갱신시킨다
#         xi = X_b[random_index:random_index + 1]
#         yi = y[random_index:random_index + 1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)  # 기울기를 구한다
#         eta = learning_schdule(epoch * m + i)  # learning rate를 계속 바꾼다.(점점 줄어듦)
#         theta = theta - eta * gradients
#         theta_path_sgd.append(theta)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
#
# plt.show()

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
# print(sgd_reg.fit(X, y.ravel()))
#
# print(sgd_reg.intercept_, sgd_reg.coef_)
m = len(X_b)

theta_path_mgd = []
n_iterations = 50
minibatch_size = 20
np.random.seed(42)
theta = np.random.randn(2, 1)  # 랜덤하게 theta값 초기화

t0, t1 = 200, 1000


def learning_scheduler(t):
    return t0 / (t + t1)


t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i + minibatch_size]
        yi = y_shuffled[i:i + minibatch_size]
        gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_scheduler(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

# plt.figure(figsize=(7, 4))
# plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="SGD")
# plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="MINI_BATCH")
# plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="BATCH")
# plt.legend(loc="upper left", fontsize=16)
# plt.xlabel(r"$\theta_0$", fontsize=16)
# plt.ylabel(r"$\theta_1$", fontsize=20)
# plt.axis([2.5, 4.5, 2.3, 3.9])
# plt.show()
# 우선 SGD의 경우 random으로 뽑은 단 하나의 점만 가지고 theta를 업데이트 시켜줌으로 진폭이 크고 정확한 수렴을 보장하지 못합니다.
# MINI BATCH의 경우 20개의 batch사이즈로 모든 점을 순환하며 theta를 업데이트 해주기 때문에 진폭이 SGD에 비하여 작고 어느정도는 정확한 수렴이 가능합니다.
# BATCH의 경우 모든 점을 다 보고 가중치를 업데이트 하므로 진폭이 작고 가장 정확히 수렴합니다.
# 2 - 01
import numpy as np
import numpy.random as rnd

np.random.seed(42)
m = 100
X = 6 * np.random.randn(m, 1)  # m,1 크기의 배열에 random하게 숫자를 채워줍니다.
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
# plt.plot(X,y,"b.")
# plt.xlabel("$x_1$", fontsize = 18)
# plt.ylabel("$x$", rotation  =0, fontsize = 18)
# plt.axis([-3,3,0,10])
# plt.show()

# 2 - 02 : X[0] 출력 확인
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
# print(X[0])
# 이 때 X는 위에서 random하게 넣은 수들 중 첫번째 값이 출력됩니다.

# 2 - 03 : X_poly[0] 출력 확인
# print(X_poly[0])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# print(lin_reg.intercept_)
# print(lin_reg.coef_)

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
# plt.plot(X, y, "b.")
# plt.plot(X_new, y_new, "r-", linewidth=2, label="prediction")
# plt.xlabel("$x_1$", rotation=0, fontsize=18)
# plt.ylabel("$y$", rotation=90, fontsize=18)
# plt.legend(loc="upper left", fontsize=14)
# plt.axis([-3, 3, 0, 10])
# plt.show()



from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()

    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    # Pipeline 실행
    # X = 6 * np.random.randn(9, 1)  # m,1 크기의 배열에 random하게 숫자를 채워줍니다.
    # y = 0.5 * X ** 2 + X + 2 + np.random.randn(9, 1)

    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

plt.plot(X, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
#save_fig("high_degree_polynomials_plot")
plt.show()

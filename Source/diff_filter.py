import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy import sparse
from scipy.sparse import csr_matrix

# 画像の読み込み
grayX = cv2.imread("image/yasai256.jpg", 0) / 255

# 画像サイズを求める
m, n = grayX.shape

# ➀周期シフトして差分を求める
grayX_v = abs(grayX[np.roll(np.arange(n), -1), :] - grayX) # 縦方向に周期シフト - 自分
grayX_h = abs(grayX[:, np.roll(np.arange(m), -1)] - grayX) # 横方向に周期シフト - 自分
grayX_ = grayX_v + grayX_h # 縦横

# ➁線形代数を使って差分を求める
D0 = -np.eye(m) + np.roll(np.eye(m), -1, axis = 0)
print(D0)

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(csr_matrix(np.eye(m)), csr_matrix(D0)) # 単位行列 ⊗ D0
Dh = sparse.kron(csr_matrix(D0),csr_matrix(np.eye(n))) # D0 ⊗ 単位行列

# 画像データ(2次元配列)からm×n行1列のベクトルデータに変換
grayX_vec = grayX.reshape(m * n, 1, order = 'F')

# エッジ強度(検出)の計算
grayX_dv = abs(Dv @ grayX_vec) # 縦
grayX_dh = abs(Dh @ grayX_vec) # 横
grayX_d = grayX_dv + grayX_dh # 縦横

# 画像データ(2次元配列)に戻す
grayX_dv_ = grayX_dv.reshape(grayX.shape, order = 'F')
grayX_dh_ = grayX_dh.reshape(grayX.shape, order = 'F')
grayX_d_ = grayX_d.reshape(grayX.shape, order = 'F')

# ➀、➁の誤差を表示
print(la.norm(grayX_v - grayX_dv_))
print(la.norm(grayX_h - grayX_dh_))
print(la.norm(grayX_ - grayX_d_))

# 結果を表示
plt.figure()
plt.imshow(grayX, cmap = "gray")
plt.title('gray')
plt.figure()
plt.imshow(grayX_v, cmap = "gray")
plt.title('1.v')
plt.figure()
plt.imshow(grayX_h, cmap = "gray")
plt.title('1.h')
plt.figure()
plt.imshow(grayX_, cmap = "gray")
plt.title('1.vh')
plt.figure()
plt.imshow(grayX_dv_, cmap = "gray")
plt.title('2.v')
plt.figure()
plt.imshow(grayX_dh_, cmap = "gray")
plt.title('2.h')
plt.figure()
plt.imshow(grayX_d_, cmap = "gray")
plt.title('2.vh')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix

# 長方形を含む四角形の画像の読み込み
X = cv2.imread("image/yasai512_256.jpg")
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
grayX = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY) / 255

# 画像サイズを求める
y, x = grayX.shape

# 1.周期シフトして差分を求める
grayX_v = abs(grayX[np.roll(np.arange(y), -1), :] - grayX) # 縦方向に周期シフト - 自分
grayX_h = abs(grayX[:, np.roll(np.arange(x), -1)] - grayX) # 横方向に周期シフト - 自分
grayX_ = grayX_v + grayX_h # 縦横

# 2.線形代数を使って差分を求める
D0_v = -np.eye(y) + np.roll(np.eye(y), -1, axis = 0)
D0_h = -np.eye(x) + np.roll(np.eye(x), -1, axis = 0)
print(D0_v)
print(D0_h)

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(coo_matrix(np.eye(x)), coo_matrix(D0_v)).tocsr() # 単位行列 ⊗ D0_v
Dh = sparse.kron(coo_matrix(np.eye(y)), coo_matrix(D0_h)).tocsr() # 単位行列 ⊗ D0_h
print(Dv)
print()
print(Dh)

# 画像データ(2次元配列)からm×n行1列のベクトルデータに変換
grayX_vec = grayX.reshape(y * x, 1, order = 'F')
grayX_vec2 = grayX.reshape(y * x, 1, order = 'C')

# エッジ強度(検出)の計算
grayX_dv = abs(Dv @ grayX_vec) # 縦
grayX_dh = abs(Dh @ grayX_vec2) # 横

# 画像データ(2次元配列)に戻す
grayX_dv_ = grayX_dv.reshape(grayX.shape, order = 'F')
grayX_dh_ = grayX_dh.reshape(grayX.shape, order = 'C')
grayX_d_ = grayX_dv_ + grayX_dh_

# 1、2 の誤差を表示
print(la.norm(grayX_v - grayX_dv_))
print(la.norm(grayX_h - grayX_dh_))
print(la.norm(grayX_ - grayX_d_))

# 結果を表示
plt.figure()
plt.imshow(X)
plt.title('original')
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
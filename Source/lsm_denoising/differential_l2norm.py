import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix

# 長方形を含む四角形の画像の読み込み
X = cv2.imread("image/yasai256.jpg")
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
grayX = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY) / 255

# 画像サイズを求める
y, x = grayX.shape

# ベクトル化するための連結方向の決定
# （これを決めないと長方形画像のとき黒線が出現する）
order = 'F' if y < x else 'C'

# ノイズの生成
sigma = 0.5; # ノイズの強さを調整
noize_X = grayX + sigma * np.random.normal(0, sigma, grayX.shape)
Xtld = noize_X.reshape(y * x, 1, order = order)

# 線形代数を使って差分を求める
D0_v = -np.eye(y) + np.roll(np.eye(y), -1, axis = 0)
D0_h = -np.eye(x) + np.roll(np.eye(x), -1, axis = 1)

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(csc_matrix(np.eye(x)), csc_matrix(D0_v)) # 単位行列 ⊗ D0_v
Dh = sparse.kron(csc_matrix(np.eye(y)), csc_matrix(D0_h)) # 単位行列 ⊗ D0_h
DvT = Dv.T
DhT = Dh.T

lam = 0.8; # λ（画像の滑らかさを考慮するパラメータ）
I = sparse.eye(y * x, y * x)

# 最小二乗法
X_star = sparse.linalg.spsolve(I + 2 * lam * (DvT @ Dv) + 2 * lam * (DhT @ Dh), Xtld)
X_star = X_star.reshape(y, x, order = order)

print(la.norm(grayX - X_star))
print(cv2.PSNR(grayX, noize_X))
print(cv2.PSNR(grayX, X_star))
    
# 結果を表示
plt.figure()
plt.imshow(X)
plt.title('original')
plt.figure()
plt.imshow(grayX, cmap = "gray")
plt.title('gray')
plt.figure()
plt.imshow(noize_X, cmap = "gray")
plt.title('noise')
plt.figure()
plt.imshow(X_star, cmap = "gray")
plt.title('denoising')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix

# 長方形を含む四角形の画像の読み込み
X = cv2.imread("image/yasai256.jpg")
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
grayX = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY) / 255

# 画像サイズを求める
y, x = grayX.shape

# ベクトル化するための連結方向の決定
# （これを決めないと長方形画像のとき黒線が出現する）
order = 'F' if y < x else 'C'

# 画像に穴を開ける(欠損画像)
phi = np.random.rand(y, x) > 0.7
loss_X = phi * grayX

# 線形代数を使って差分を求める
D0_v = -np.eye(y) + np.roll(np.eye(y), -1, axis = 0)
D0_h = -np.eye(x) + np.roll(np.eye(x), -1, axis = 1)

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(coo_matrix(np.eye(x)), coo_matrix(D0_v)).tocsr() # 単位行列 ⊗ D0_v
Dh = sparse.kron(coo_matrix(np.eye(y)), coo_matrix(D0_h)).tocsr() # 単位行列 ⊗ D0_h
DvT = Dv.T
DhT = Dh.T

lam = 0.5 # λ（画像の滑らかさを考慮するパラメータ）
I = sparse.eye(y * x, y * x)

# 最小二乗法（微分係数 = 0 で解く）
phi_vec = sparse.coo_matrix((phi.flatten(), (np.arange(y * x), np.arange(y * x))), shape=(y * x, y * x)).tocsr()
loss_X_vec = loss_X.flatten()
X_star = sparse.linalg.spsolve(phi_vec.T @ phi_vec + 2 * lam * (Dv.T @ Dv) + 2 * lam * (Dh.T @ Dh), phi_vec.T @ loss_X_vec)
X_star = X_star.reshape(y, x, order = order)

print(la.norm(grayX - X_star))
print(cv2.PSNR(grayX, loss_X))
print(cv2.PSNR(grayX, X_star))
    
# 結果を表示
plt.figure()
plt.imshow(X)
plt.title('original')
plt.figure()
plt.imshow(grayX, cmap = "gray")
plt.title('gray')
plt.figure()
plt.imshow(loss_X, cmap = "gray")
plt.title('loss')
plt.figure()
plt.imshow(X_star, cmap = "gray")
plt.title('interpolation')
plt.show()
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

# 画像に穴を開ける(欠損画像)
phi = np.random.rand(y, x) > 0.3
loss_X = phi * grayX

# 線形代数を使って差分を求める
D0_v = -np.eye(y) + np.roll(np.eye(y), -1, axis = 0) # y×y
D0_h = -np.eye(x) + np.roll(np.eye(x), -1, axis = 0) # x×x

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(coo_matrix(np.eye(x)), coo_matrix(D0_v)).tocsr() # 単位行列 ⊗ D0_v（yx×yx）
Dh = sparse.kron(coo_matrix(D0_h), coo_matrix(np.eye(y))).tocsr() # 単位行列 ⊗ D0_h（yx×yx）

lam = 0.5 # λ（画像の滑らかさを考慮するパラメータ）

# 最小二乗法（微分係数 = 0 で解く）
phi = phi.astype(int)
phi_vec = sparse.diags(phi.reshape(-1), offsets = 0)
loss_X_vec = loss_X.reshape(y * x, 1)
# X_star, info = sparse.linalg.cg(phi_vec.T @ phi_vec + 2 * lam * (Dv.T @ Dv) + 2 * lam * (Dh.T @ Dh), phi_vec.T @ loss_X_vec)
# X_star = X_star.reshape(y, x)

# 最小二乗法（最急降下法で解く）
lam = 1
alpha = 0.1
maxIter = 100
xcurr = sparse.csr_matrix((y * x, 1))
for i in range(maxIter):
    xnext = phi_vec.T @ (phi_vec @ xcurr - loss_X_vec) + 2 * lam * (Dv.T @ Dv @ xcurr) + 2 * lam * (Dh.T @ Dh @ xcurr)
    xcurr -= alpha * xnext

xcurr = xcurr.reshape(y, x)

print(f"欠損画像のPSNR:{cv2.PSNR(grayX, loss_X)}")
print(f"最急降下法の推定画像のPSNR:{cv2.PSNR(grayX, xcurr)}")
# print(f"微分係数=0の推定画像のPSNR:{cv2.PSNR(grayX, X_star)}")

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
plt.imshow(xcurr, cmap = "gray")
plt.title('gd_interpolation')
# plt.figure()
# plt.imshow(X_star, cmap = "gray")
# plt.title('differential_interpolation')
plt.show()
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
order = 'F'

# 画像に穴を開ける(欠損画像)
phi = np.random.rand(y, x) > 0.9
phi = phi.astype(int)
phi_vec = sparse.diags(phi.reshape(-1,order = 'F'), offsets = 0)

loss_X = phi * grayX
loss_X_vec = loss_X.reshape(y * x, 1, order = 'F')


# 線形代数を使って差分を求める
D0_v = -np.eye(y) + np.roll(np.eye(y), -1, axis = 0) # y×y
D0_h = -np.eye(x) + np.roll(np.eye(x), -1, axis = 0) # x×x

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(coo_matrix(np.eye(x)), coo_matrix(D0_v)).tocsr() # 単位行列 ⊗ D0_v（yx×yx）
Dh = sparse.kron(coo_matrix(D0_h), coo_matrix(np.eye(y))).tocsr() # 単位行列 ⊗ D0_h（yx×yx）
D = sparse.vstack([Dv, Dh],format = 'csr')
DT = D.T
dy, dx = D.shape
DTD = D.T @ D

# 初期化
lam = 0.0001 # λ（画像の滑らかさを考慮するパラメータ）
rho = 0.1 # ステップ幅

xcurr = np.zeros((y * x, 1))
zcurr = np.zeros((dy, 1))
ucurr = np.zeros((dy, 1))

# ADMM
# x^(k + 1) = (Φ^TΦ + ρ(D^T)D)^(-1)((Φ^T)loss_X + ρ(D^T)(z^(k) - u^(k))) 
# z^(k + 1) = (ρ/(2λ + ρ))(Dx^(k + 1) + u^(k)) 
# u^(k + 1) = u^(k) + (Dx^(k + 1) + z^(k + 1))
maxIter = 500
epsilon = 1e-13
A = phi_vec.T @ phi_vec + rho * (D.T @ D)

for k in range(maxIter):
    xnext, info = sparse.linalg.cg(A, phi_vec.T @ loss_X_vec + rho * (D.T) @ (zcurr - ucurr))
    xnext = xnext.reshape(-1, 1, order = order)
    znext = rho /(2 * lam + rho) * (D @ xnext + ucurr)
    unext = ucurr + ((D @ xnext) - znext)
    diff = la.norm(xnext - xcurr)
    if diff > epsilon:
        print(f"iter:{k}, diff:{diff:.40f}")
        xcurr = xnext
        zcurr = znext
        ucurr = unext
    else:
        break

xcurr = xcurr.reshape(y, x, order = order)
print(la.norm(grayX - xcurr))
print(cv2.PSNR(grayX, loss_X))
print(cv2.PSNR(grayX, xcurr))
    
# 結果を表示
plt.figure()
plt.imshow(X)
plt.title('original')
plt.figure()
plt.imshow(grayX, cmap = "gray")
plt.title('gray')
plt.figure()
plt.imshow(loss_X, cmap = "gray")
plt.title('noise')
plt.figure()
plt.imshow(xcurr, cmap = "gray")
plt.title('interpolation')
plt.show()
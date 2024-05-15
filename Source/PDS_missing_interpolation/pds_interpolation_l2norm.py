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
phi = np.random.rand(y, x) > 0.5
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
D = sparse.vstack([Dv, Dh], format = 'csr')
DT = D.T
dy, dx = D.shape

# PDS
# xtld^{(n)} = x^{(n)} - γ_1(D^T)z^{(n)}
# x^{(n + 1)} = (xtld^{(n)} + 2γ_1b) / (1 + 2γ_1)
# ztld^{(n)} = z^{(n)} - γ_2(D^T)(2x^{(n + 1)} - x^{(n)})
# z^{(n + 1)} = ztld^{(n)} - γ_2 ((1 / γ_2)z / (1 + 2(1 / γ_2)lambda))
xcurr = np.zeros((y * x, 1))
zcurr = np.zeros((3 * y * x, 1))
L = sparse.vstack([phi_vec,D], format = 'csr')
LT = L.T

gamma1 = 0.25
gamma2 = 0.25
lam = 0.1 # λ（画像の滑らかさを考慮するパラメータ）

def proxL2norm(z, b,gamma):
    return (z + gamma * b) / (1 + gamma)

maxIter = 500
epsilon = 1e-13
for k in range(maxIter):
    xtld = xcurr - gamma1 * LT @ zcurr
    xnext = xtld
    ztld = zcurr + gamma2 * L @ (2 * xnext - xcurr)
    ztld1 = ztld[:len(xcurr)]
    ztld2 = ztld[len(xcurr):]
    znext1 = ztld1 - gamma2 * proxL2norm((1 / gamma2) * ztld1, loss_X_vec, 1 / gamma2)
    znext2 = ztld2 - gamma2 * proxL2norm((1 / gamma2) * ztld2, 0, (1 / gamma2) * 2 * lam)
    znext = np.vstack([znext1, znext2])
    diff = la.norm(znext - zcurr)
    if diff > epsilon:
        print(f"iter:{k}, diff:{diff:.40f}")
        xcurr = xnext
        zcurr = znext
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
plt.title('loss')
plt.figure()
plt.imshow(xcurr, cmap = "gray")
plt.title('pds l2 interpolation')
plt.show()
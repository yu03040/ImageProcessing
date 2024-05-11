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
order = 'F'

# ノイズの生成
sigma = 0.3; # ノイズの強さを調整
noise_X = grayX + sigma * np.random.normal(0, sigma, grayX.shape)
noise_X_vec = noise_X.reshape(y * x, 1, order = order)

# 線形代数を使って差分を求める
D0_v = -np.eye(y) + np.roll(np.eye(y), -1, axis = 0) # y×y
D0_h = -np.eye(x) + np.roll(np.eye(x), -1, axis = 0) # x×x

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(coo_matrix(np.eye(x)), coo_matrix(D0_v)).tocsr() # 単位行列 ⊗ D0_v（yx×yx）
Dh = sparse.kron(coo_matrix(D0_h), coo_matrix(np.eye(y))).tocsr() # 単位行列 ⊗ D0_h（yx×yx）
D = sparse.vstack([Dv, Dh],format = 'csr')
DT = D.T
dy, dx = D.shape

# PDS
# xtld^{(n)} = x^{(n)} - γ_1(D^T)z^{(n)}
# x^{(n + 1)} = (xtld^{(n)} + 2γ_1b) / (1 + 2γ_1)
# ztld^{(n)} = z^{(n)} - γ_2(D^T)(2x^{(n + 1)} - x^{(n)})
# z^{(n + 1)} = ztld^{(n)} - γ_2 ((1 / γ_2)z / (1 + 2(1 / γ_2)lambda))
xcurr = np.zeros((y * x, 1))
zcurr = np.zeros((dy, 1))

gamma1 = 0.5
gamma2 = 0.5
lam = 0.5 # λ（画像の滑らかさを考慮するパラメータ）
def proxL2norm(z, b,gamma):
    return (z + gamma * b) / (1 + gamma)

maxIter = 500
epsilon = 1e-13
for k in range(maxIter):
    xtld = xcurr - gamma1 * DT @ zcurr
    xnext = proxL2norm(xtld, noise_X_vec, gamma1)
    ztld = zcurr + gamma2 * D @ (2 * xnext - xcurr)
    znext = ztld - gamma2 * proxL2norm((1 / gamma2) * ztld, 0, 1 / gamma2 * 2 * lam)
    diff = la.norm(xnext - xcurr)
    if diff > epsilon:
        print(f"iter:{k}, diff:{diff:.40f}")
        xcurr = xnext
        zcurr = znext
    else:
        break

xcurr = xcurr.reshape(y, x, order = order)
print(la.norm(grayX - xcurr))
print(cv2.PSNR(grayX, noise_X))
print(cv2.PSNR(grayX, xcurr))
    
# 結果を表示
plt.figure()
plt.imshow(X)
plt.title('original')
plt.figure()
plt.imshow(grayX, cmap = "gray")
plt.title('gray')
plt.figure()
plt.imshow(noise_X, cmap = "gray")
plt.title('noise')
plt.figure()
plt.imshow(xcurr, cmap = "gray")
plt.title('pds l2 denoising')
plt.show()
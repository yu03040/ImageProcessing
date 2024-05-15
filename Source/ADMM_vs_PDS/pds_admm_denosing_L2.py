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

# ノイズの生成
sigma = 0.1; # ノイズの強さを調整
noise_X = grayX + sigma * np.random.randn(y, x)
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

# 単位行列
I = sparse.eye(y * x, y * x)

lam = 0.5
rho = 0.1
gamma1 = 0.1
gamma2 = 0.1
epsilon = 1e-13

# 最小二乗法（微分係数 = 0 で厳密解を求める
xstar, info1 = sparse.linalg.cg(I + 2 * lam * (D.T @ D), noise_X_vec)
xstar = xstar.reshape(y, x, order = order)

# ADMM
x1curr = np.random.rand(y * x, 1)
z1curr = np.random.rand(dy, 1)
u1curr = np.random.rand(dy, 1)

maxiter1 = 500
A = I + rho * (D.T @ D)

for i in range(maxiter1):
    x1next, info2 = sparse.linalg.cg(A, noise_X_vec + rho * DT @ (z1curr - u1curr))
    x1next = x1next.reshape(-1, 1, order = order)
    z1next = rho /(2 * lam + rho) * (D @ x1next + u1curr)
    z1next = z1next.reshape(-1, 1, order = order)
    u1next = u1curr + ((D @ x1next) - z1next)
    difftemp = la.norm(x1next - x1curr)
    if difftemp > epsilon:
        diff1 = difftemp
        print(f"iter:{i}, diff:{diff1:.40f}")
        x1curr = x1next
        z1curr = z1next
        u1curr = u1next
    else:
        i-=1
        break

x1curr = x1curr.reshape(y, x, order = order)

# PDS
x2curr = np.random.rand(y * x, 1)
z2curr = np.random.rand(dy, 1)

def proxL2norm(z, b,gamma):
    return (z + gamma * b) / (1 + gamma)

maxiter2 = 500
for k in range(maxiter2):
    xtld = x2curr - gamma1 * DT @ z2curr
    x2next = proxL2norm(xtld, noise_X_vec, gamma1)
    ztld = z2curr + gamma2 * D @ (2 * x2next - x2curr)
    z2next = ztld - gamma2 * proxL2norm((1 / gamma2) * ztld, 0, 1 / gamma2 * 2 * lam)
    difftemp = la.norm(x2next - x2curr)
    if difftemp > epsilon:
        diff2 = difftemp
        print(f"iter:{k}, diff:{diff2:.40f}")
        x2curr = x2next
        z2curr = z2next
    else:
        k-=1
        break

x2curr = x2curr.reshape(y, x, order = order)
    
# 結果を表示
print(f"元画像と admm の誤差：{la.norm(grayX - x1curr)}")
print(f"元画像と pds  の誤差：{la.norm(grayX - x2curr)}")
print(f"ノイズと admm の誤差：{la.norm(noise_X - x1curr)}")
print(f"ノイズと pds  の誤差：{la.norm(noise_X - x2curr)}")
print(f"厳密解と admm の誤差：{la.norm(xstar - x1curr)}")
print(f"厳密解と pds  の誤差：{la.norm(xstar - x2curr)}")
print(f"admm と pds  の誤差：{la.norm(x1curr - x2curr)}")
print(f"ノイズ の PSNR：{cv2.PSNR(grayX, noise_X)}")
print(f"admm  の PSNR：{cv2.PSNR(grayX, x1curr)}")
print(f"pds   の PSNR：{cv2.PSNR(grayX, x2curr)}")
print(f"admm 反復回数：{i}, 現在の解と更新した解の誤差：{diff1:.40f}")
print(f"pds  反復回数：{k}, 現在の解と更新した解の誤差：{diff2:.40f}")
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
plt.imshow(xstar, cmap = "gray")
plt.title('exact solution denoising')
plt.figure()
plt.imshow(x1curr, cmap = "gray")
plt.title('admm l2 denoising')
plt.figure()
plt.imshow(x2curr, cmap = "gray")
plt.title('pds l2 denoising')
plt.show()
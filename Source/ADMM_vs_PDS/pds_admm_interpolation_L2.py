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
phi = np.random.rand(y, x) > 0.3
phi = phi.astype(int)
phi_vec = sparse.diags(phi.reshape(-1,order = order), offsets = 0)
loss_X = phi * grayX
loss_X_vec = loss_X.reshape(y * x, 1, order = order)

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

lam = 0.25
rho = 0.1
gamma1 = 0.1
gamma2 = 0.1
epsilon = 1e-13

# 最小二乗法（微分係数 = 0 で厳密解を求める）
A0 = phi_vec.T @ phi_vec + 2 * lam * (D.T @ D)
xstar, info = sparse.linalg.cg(A0, phi_vec.T @ loss_X_vec)
xstar = xstar.reshape(y, x, order = order)

# ADMM
x1curr = np.random.rand(y * x, 1)
z1curr = np.random.rand(dy, 1)
u1curr = np.random.rand(dy, 1)

maxiter1 = 500
A1 = phi_vec.T @ phi_vec + rho * (D.T @ D)

for i in range(maxiter1):
    x1next, info = sparse.linalg.cg(A1, phi_vec.T @ loss_X_vec + rho * (D.T) @ (z1curr - u1curr))
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
z2curr = np.random.rand(3 * y * x, 1)
L = sparse.vstack([phi_vec,D], format = 'csr')
LT = L.T

def proxL2norm(z, b,gamma):
    return (z + gamma * b) / (1 + gamma)

maxiter2 = 500
for k in range(maxiter2):
    xtld = x2curr - gamma1 * LT @ z2curr
    x2next = xtld
    ztld = z2curr + gamma2 * L @ (2 * x2next - x2curr)
    ztld1 = ztld[:len(x2curr)]
    ztld2 = ztld[len(x2curr):]
    z2next1 = ztld1 - gamma2 * proxL2norm((1 / gamma2) * ztld1, loss_X_vec, 1 / gamma2)
    z2next2 = ztld2 - gamma2 * proxL2norm((1 / gamma2) * ztld2, 0, (1 / gamma2) * 2 * lam)
    z2next = np.vstack([z2next1, z2next2])
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
print(f"欠損  と admm の誤差：{la.norm(loss_X - x1curr)}")
print(f"欠損  と pds  の誤差：{la.norm(loss_X - x2curr)}")
print(f"厳密解と admm の誤差：{la.norm(xstar - x1curr)}")
print(f"厳密解と pds  の誤差：{la.norm(xstar - x2curr)}")
print(f"admm  と pds  の誤差：{la.norm(x1curr - x2curr)}")
print(f"欠損   の PSNR：{cv2.PSNR(grayX, loss_X)}")
print(f"厳密解 の PSNR：{cv2.PSNR(grayX, xstar)}")
print(f"admm   の PSNR：{cv2.PSNR(grayX, x1curr)}")
print(f"pds    の PSNR：{cv2.PSNR(grayX, x2curr)}")
print(f"admm 反復回数：{i}, 現在の解と更新した解の誤差：{diff1:.40f}")
print(f"pds  反復回数：{k}, 現在の解と更新した解の誤差：{diff2:.40f}")
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
plt.imshow(xstar, cmap = "gray")
plt.title('exact solution interpolation')
plt.figure()
plt.imshow(x1curr, cmap = "gray")
plt.title('admm l2 interpolation')
plt.figure()
plt.imshow(x2curr, cmap = "gray")
plt.title('pds l2 interpolation')
plt.show()
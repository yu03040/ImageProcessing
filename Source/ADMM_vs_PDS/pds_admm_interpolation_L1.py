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
phi = np.random.rand(y * x, 1) > 0.5
phi = phi.astype(int)
phi_vec = sparse.diags(phi.reshape(-1,order = order), offsets = 0)
loss_X = phi * grayX.reshape(y * x, 1, order = order)

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

lam = 0.05
rho = 0.07
gamma1 = 0.07
gamma2 = 0.07
maxiter = 500
epsilon = 1e-2

# ADMM
x1curr = np.random.rand(y * x, 1)
z1curr = np.random.rand(dy, 1)
u1curr = np.random.rand(dy, 1)

# 軟判定しきい値関数 ->
# しきい値を持ち、入力の大きさがそのしきい値を超えないと反応しない。
# そのしきい値を超えると、しきい値分だけ縮こまった値を出力する。
def soft_threshold(y, alpha):
    return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)

A1 = phi_vec.T @ phi_vec + rho * (D.T @ D)

for i in range(maxiter):
    x1next, info = sparse.linalg.cg(A1, phi_vec.T @ loss_X + rho * (D.T) @ (z1curr - u1curr))
    x1next = x1next.reshape(-1, 1, order = order)
    z1next = soft_threshold(D @ x1next + u1curr, lam / rho).reshape(-1, 1, order = order)
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
L = sparse.vstack([phi_vec,D])
LT = L.T

def proxL2norm(z, b,gamma):
    return (z + gamma * b) / (1 + gamma)

for k in range(maxiter):
    xtld = x2curr - gamma1 * LT @ z2curr
    x2next = xtld
    ztld = z2curr + gamma2 * L @ (2 * x2next - x2curr)
    ztld1 = ztld[:len(x2curr)]
    ztld2 = ztld[len(x2curr):]
    z2next1 = ztld1 - gamma2 * proxL2norm((ztld1 / gamma2), loss_X, 1 / gamma2)
    z2next2 = ztld2 - gamma2 * soft_threshold((ztld2 / gamma2), (lam / gamma2))
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

loss_X = loss_X.reshape(y, x, order = order)    
# 結果を表示
print(f"元画像と admm の誤差：{la.norm(grayX - x1curr)}")
print(f"元画像と pds  の誤差：{la.norm(grayX - x2curr)}")
print(f"欠損 と admm  の誤差：{la.norm(loss_X - x1curr)}")
print(f"欠損 と pds   の誤差：{la.norm(loss_X - x2curr)}")
print(f"admm と pds   の誤差：{la.norm(x1curr - x2curr)}")
print(f"欠損 の PSNR：{cv2.PSNR(grayX, loss_X)}")
print(f"admm の PSNR：{cv2.PSNR(grayX, x1curr)}")
print(f"pds  の PSNR：{cv2.PSNR(grayX, x2curr)}")
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
plt.imshow(x1curr, cmap = "gray")
plt.title('admm l1 interpolation')
plt.figure()
plt.imshow(x2curr, cmap = "gray")
plt.title('pds l1 interpolation')
plt.show()
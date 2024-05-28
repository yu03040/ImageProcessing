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
temp_grayX = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY) / 255

# 画像サイズを求める
y, x = grayX.shape

# ベクトル化するための連結方向の決定
order = 'F'

# ノイズの生成
sigma = 0.1; # ノイズの強さを調整
noise_X = grayX + sigma * np.random.randn(y, x)
Xtld = noise_X.reshape(y * x, 1, order = order)

# 線形代数を使って差分を求める
D0_v = -np.eye(y) + np.roll(np.eye(y), -1, axis = 0)
D0_h = -np.eye(x) + np.roll(np.eye(x), -1, axis = 0)

# クロネッカー積を使ってフィルタ係数を求める
Dv = sparse.kron(coo_matrix(np.eye(x)), coo_matrix(D0_v)).tocsr() # 単位行列 ⊗ D0_v
Dh = sparse.kron(coo_matrix(D0_h), coo_matrix(np.eye(y))).tocsr() # 単位行列 ⊗ D0_h
DvT = Dv.T
DhT = Dh.T

# λを設定（画像の滑らかさを考慮するパラメータ）
lam = 0.5

# 最小二乗法
I = sparse.eye(y * x, y * x)
X_star = sparse.linalg.spsolve(I + 2 * lam * (DvT @ Dv) + 2 * lam * (DhT @ Dh), Xtld)
X_star = X_star.reshape(y, x, order = order)

# 最急降下法（勾配降下法）
alpha = 0.1 # ステップ幅(α)
xcurr = np.zeros((y * x, 1))
diff = 0.0
maxIter = 100
for i in range(maxIter):
    xnext = xcurr - alpha * ((xcurr - Xtld) + 2 * lam * (DvT @ Dv @ xcurr) + 2 * lam * (DhT @ Dh @ xcurr))
    difftemp = la.norm(xnext - xcurr)
    if difftemp > 1e-3:
        diff = difftemp
        print(f"iter:{i}, diff:{diff:.40f}")
        xcurr = xnext
    else:
        i-=1
        break

xcurr = xcurr.reshape(y, x, order = order)
print(f"ノイズ画像のPSNR:{cv2.PSNR(grayX, noise_X)}")
print(f"微分係数=0の推定画像のPSNR:{cv2.PSNR(grayX, X_star)}")
print(f"最急降下法の推定画像のPSNR:{cv2.PSNR(grayX, xcurr)}")
print(f"最小二乗法と最急降下法の差:{la.norm(X_star - xcurr)}")
print(f"最急降下法の反復回数：{i}, 現在の解と更新した解の誤差：{diff:.40f}")

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
plt.imshow(X_star, cmap = "gray")
plt.title('exactsolution_denoising')
plt.figure()
plt.imshow(xcurr, cmap = "gray")
plt.title('gd_denoising')
plt.show()
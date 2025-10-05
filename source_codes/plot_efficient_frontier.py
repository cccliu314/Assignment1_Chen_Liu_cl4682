# -*- coding: utf-8 -*-
"""
Minimal Efficient Frontier plotter (no pypfopt, no cvx/cvxopt).
Inputs:  source_codes/portfolio_output/all_return_table.pickle
         source_codes/portfolio_output/all_stocks_info.pickle
Output:  source_codes/portfolio_output/efficient_frontier.png
Usage:   conda activate MyNewEnv  (or your env with numpy/pandas/matplotlib)
         python source_codes/plot_frontier_minimal.py
"""
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------- robust path ---------------------
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE / "portfolio_output",
    HERE.parent / "source_codes" / "portfolio_output",
    HERE.parent / "portfolio_output",
]
data_dir = None
for p in CANDIDATES:
    if (p / "all_return_table.pickle").exists():
        data_dir = p
        break
if data_dir is None:
    raise FileNotFoundError("Cannot find portfolio_output folder with pickles. "
                            f"Tried: {CANDIDATES}")

print(f"[info] using data_dir = {data_dir}")

# --------------------- load pickles ---------------------
with open(data_dir / "all_return_table.pickle", "rb") as f:
    all_return_table = pickle.load(f)

# all_stocks_info 目前不强依赖，但加载以验证文件齐全
if (data_dir / "all_stocks_info.pickle").exists():
    with open(data_dir / "all_stocks_info.pickle", "rb") as f:
        all_stocks_info = pickle.load(f)

# --------------------- pick latest trade_date, build return matrix R ---------------------
trade_dates = sorted(all_return_table.keys())
if len(trade_dates) == 0:
    raise RuntimeError("all_return_table.pickle seems empty.")
td = trade_dates[-1]
print(f"[info] chosen trade_date = {td}")

ret_long = all_return_table[td].copy()

# 统一列名：ID / datadate / daily_return
col_map = {c.lower(): c for c in ret_long.columns}
def col_like(*names):
    for n in names:
        if n in col_map: return col_map[n]
    return None

id_col = col_like("id") or col_like("tic") or col_like("gvkey")
dt_col = col_like("datadate") or col_like("date")
ret_col = col_like("daily_return") or col_like("return") or col_like("ret")

assert id_col is not None,  "Cannot find ID/tic/gvkey column in ret_long."
assert dt_col is not None,  "Cannot find datadate/date column in ret_long."
assert ret_col is not None, "Cannot find daily_return/return column in ret_long."

ret_long[dt_col] = pd.to_datetime(ret_long[dt_col])
ret_long[id_col] = ret_long[id_col].astype(str)

# 只保留必要列，透视成 (date x asset) 的收益矩阵
R = ret_long.pivot(index=dt_col, columns=id_col, values=ret_col).sort_index()

# 丢掉有缺口的资产以避免协方差出 NaN（也可用 fillna 再 drop）
R = R.dropna(axis=1, how="any")
if R.shape[1] < 2:
    raise RuntimeError(f"Not enough assets after cleaning. R shape = {R.shape}")

print(f"[info] R shape (dates x assets) = {R.shape}")

# --------------------- annualize stats ---------------------
TRADING_DAYS = 252.0
mu = R.mean() * TRADING_DAYS                # 年化期望收益 (Series, len = n_assets)
S  = R.cov()  * TRADING_DAYS                # 年化协方差   (DataFrame n x n)

# --------------------- random portfolios (Dirichlet) ---------------------
n_assets = R.shape[1]
n_samples = 12000

rng = np.random.default_rng(7)
W = rng.dirichlet(np.ones(n_assets), size=n_samples)   # 每行和为 1，且非负

# 组合收益/波动
mu_vec = mu.values
S_mat  = S.values

port_ret  = W @ mu_vec
# 方差 = w^T S w
port_var  = np.einsum('ij,jk,ik->i', W, S_mat, W)
port_risk = np.sqrt(port_var)

# 无风险利率（可改）
rf = 0.02
sharpe = (port_ret - rf) / np.maximum(port_risk, 1e-12)

# 标记：最小方差 & 最大 Sharpe
idx_min_var   = np.argmin(port_risk)
idx_max_sharp = np.argmax(sharpe)

# --------------------- plot ---------------------
plt.figure(figsize=(9,6))
sc = plt.scatter(port_risk, port_ret, s=6, alpha=0.35, c=sharpe, cmap="viridis")
plt.colorbar(sc, label="Sharpe")

plt.scatter(port_risk[idx_min_var], port_ret[idx_min_var],
            color="red", s=80, marker="x", label="Min Variance")
plt.scatter(port_risk[idx_max_sharp], port_ret[idx_max_sharp],
            color="orange", s=80, marker="o", edgecolor="k", label="Max Sharpe")

plt.title(f"Efficient Frontier (approx.) — {td}")
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_png = data_dir / "efficient_frontier.png"
plt.savefig(out_png, dpi=300)
print(f"[ok] saved: {out_png}")
plt.show()

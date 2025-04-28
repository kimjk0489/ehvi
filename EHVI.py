import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.optim import optimize_acqf

st.set_page_config(page_title="Slurry 조성 최적화 (qEHVI)", layout="wide")
st.title("Slurry 조성 최적화: qEHVI 기반 Bayesian Optimization")

# CSV 파일 불러오기
CSV_PATH = "Slurry_data_LHS.csv"
df = pd.read_csv(CSV_PATH)

x_cols = ["carbon_black_wt%", "CMC_wt%", "SBR_wt%", "solvent_wt%", "graphite_wt%"]
y_cols = ["yield_stress", "viscosity"]

X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 입력값 스케일링
param_bounds = {
    "carbon_black_wt%": (1, 5.0),
    "CMC_wt%": (0.5, 2),
    "SBR_wt%": (1, 4.0),
    "solvent_wt%": (60.0, 80.0),
    "graphite_wt%": (20.0, 40.0),
}
bounds_array = np.array([param_bounds[k] for k in x_cols])
x_scaler = MinMaxScaler()
x_scaler.fit(bounds_array.T)
X_scaled = x_scaler.transform(X_raw)

train_x_base = torch.tensor(X_scaled, dtype=torch.double)
train_y_base = torch.tensor(Y_raw, dtype=torch.double)

# MultiTask 학습 데이터 만들기
train_X = torch.cat([
    torch.cat([train_x_base, torch.full_like(train_x_base[:, :1], i)], dim=1)
    for i in range(train_y_base.shape[1])
], dim=0)
train_Y = train_y_base.T.reshape(-1, 1)
train_Yvar = torch.full_like(train_Y, 1e-6)

# 모델 학습
model = MultiTaskGP(train_X, train_Y, task_feature=-1, train_Yvar=train_Yvar, rank=1)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# --- 방향성 반영: viscosity는 낮을수록 좋음
train_y_base_hv = train_y_base.clone()
train_y_base_hv[:, 1] = -train_y_base_hv[:, 1]  # viscosity 방향 뒤집기

# ref_point도 맞춰서 (yield_stress min-1, viscosity min-1)
ref_point = [
    float(train_y_base_hv[:, 0].min()) - 1,
    float(train_y_base_hv[:, 1].min()) - 1,
]

# Nondominated Partitioning
partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point), Y=train_y_base_hv)

# qEHVI Acquisition Function
acq_func = qExpectedHypervolumeImprovement(model=model, ref_point=ref_point, partitioning=partitioning)

# 최적화하여 새로운 추천 조성 찾기
bounds = torch.tensor([[0.0] * len(x_cols), [1.0] * len(x_cols)], dtype=torch.double)
candidate_scaled, _ = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=128,
)
candidate_wt = x_scaler.inverse_transform(candidate_scaled.detach().cpu().numpy())[0]
candidate_wt = candidate_wt / np.sum(candidate_wt) * 100  # wt%로 정규화

# 추천된 조성 출력
st.subheader("추천된 조성 (qEHVI 최적화)")
for col in x_cols:
    idx = x_cols.index(col)
    st.write(f"{col}: **{candidate_wt[idx]:.2f} wt%**")
st.write(f"**총합**: {np.sum(candidate_wt):.2f} wt%")

# 추천 조성의 예측 결과 출력
X_predict = x_scaler.transform(candidate_wt.reshape(1, -1))
X_mt = torch.cat([
    torch.cat([torch.tensor(X_predict, dtype=torch.double), torch.tensor([[0.0]])], dim=1),
    torch.cat([torch.tensor(X_predict, dtype=torch.double), torch.tensor([[1.0]])], dim=1)
], dim=0)
posterior = model.posterior(X_mt)
yield_pred = posterior.mean[0].item()
visc_pred = posterior.mean[1].item()
st.write(f"**예측 Yield Stress**: {yield_pred:.2f} Pa")
st.write(f"**예측 Viscosity**: {visc_pred:.3f} Pa.s")

# --- Pareto Front 시각화 ---
Y_invert = train_y_base.clone()
Y_invert[:, 1] = -Y_invert[:, 1]  # viscosity 낮을수록 좋음 반영
pareto_mask = is_non_dominated(Y_invert)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(train_y_base[:, 1], train_y_base[:, 0], color='gray', label='Data')
ax.scatter(train_y_base[pareto_mask, 1], train_y_base[pareto_mask, 0], color='red', label='Pareto Front')
ax.scatter(visc_pred, yield_pred, color='blue', s=100, label='Candidate')
ax.set_xlabel("Viscosity [Pa.s]")
ax.set_ylabel("Yield Stress [Pa]")
ax.set_title("Pareto Front: Yield Stress ↑, Viscosity ↓")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Hypervolume 전체 Progress 재계산 ---
hv_log_path = "hv_tracking.csv"

hv_list = []
for i in range(1, len(train_y_base) + 1):  # i=1부터
    current_Y = train_y_base[:i]
    current_Y_hv = current_Y.clone()
    current_Y_hv[:, 1] = -current_Y_hv[:, 1]  # viscosity 방향 반영

    ref_point_i = [
        float(current_Y_hv[:, 0].min()) - 1,
        float(current_Y_hv[:, 1].min()) - 1,
    ]

    bd = DominatedPartitioning(ref_point=torch.tensor(ref_point_i), Y=current_Y_hv)
    hv = bd.compute_hypervolume().item()
    hv_list.append({"iteration": i, "hv": hv})  # iteration은 1부터

hv_df = pd.DataFrame(hv_list)
hv_df.to_csv(hv_log_path, index=False)

# Hypervolume Progress 시각화
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(hv_df["iteration"], hv_df["hv"], marker='o')
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Hypervolume")
ax2.set_title("Hypervolume Progress Over Iterations")
ax2.set_xticks(np.arange(1, hv_df["iteration"].max() + 1, 1))  # 🔥 1부터 최대 iteration까지 1단위로 찍기
ax2.grid(True)
st.pyplot(fig2)

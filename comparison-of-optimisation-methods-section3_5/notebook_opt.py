#!/usr/bin/env python3


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import cvxpy as cp

# ─────────────────────────────────────────────────────────────────────────────
# SOLVER AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _pick_solver():
    available = cp.installed_solvers()
    for s in ["GUROBI", "HIGHS", "CLARABEL"]:
        if s in available:
            return s
    raise RuntimeError("No suitable solver found. Install HiGHS: pip install highspy")

SOLVER = _pick_solver()
print(f"[info] Using solver: {SOLVER}")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(0)

LOG_MU    = 4.8      # log-mean of demand distribution
LOG_SIGMA = 0.20     # log-std  of demand distribution

C_SHORT   = 20.0     # underage (shortage) cost per unit
C_HOLD    =  3.0     # overage  (holding)  cost per unit
CR        = C_SHORT / (C_SHORT + C_HOLD)   # critical ratio ≈ 0.870

TRAIN_SEED = 42      # seed for training/scenario samples (matches section44/55)
TEST_SEED  = 99      # seed for out-of-sample evaluation
N_TEST     = 2000    # out-of-sample evaluation draws

SAMPLE_SIZES = [20, 50, 100, 500]

# ─────────────────────────────────────────────────────────────────────────────
# ORACLE SOLUTION (true distribution known)
# ─────────────────────────────────────────────────────────────────────────────

def oracle_solution():
    """Newsvendor optimal under the true LogNormal distribution: F^{-1}(CR)."""
    dist = stats.lognorm(s=LOG_SIGMA, scale=np.exp(LOG_MU))
    return float(dist.ppf(CR))

ORACLE = oracle_solution()

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def sample_demand(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=LOG_MU, sigma=LOG_SIGMA, size=n)

TEST_DEMANDS = sample_demand(N_TEST, seed=TEST_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION MODELS
# ─────────────────────────────────────────────────────────────────────────────

def solve_sp(scenarios: np.ndarray) -> float:
    """
    Stochastic Programming / SAA newsvendor.
    min (1/N) Σ [C_SHORT·s_i + C_HOLD·e_i]
    s.t.  s_i ≥ d_i - x,  e_i ≥ x - d_i,  s_i,e_i,x ≥ 0
    """
    N = len(scenarios)
    x = cp.Variable(nonneg=True)
    s = cp.Variable(N, nonneg=True)
    e = cp.Variable(N, nonneg=True)
    obj  = (1.0 / N) * (C_SHORT * cp.sum(s) + C_HOLD * cp.sum(e))
    cons = [s >= scenarios - x, e >= x - scenarios]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=SOLVER, verbose=False)
    return float(x.value) if x.value is not None else np.nan


def solve_ro(scenarios: np.ndarray, gamma: float = 1.0) -> float:
    """
    Robust Optimisation with box uncertainty set.
    Uncertainty set: U = [d_min, d_max] (interval derived from scenarios).
    Budget parameter Γ controls conservatism (γ=0 → nominal, γ=1 → full robust).
    min t
    s.t. t ≥ C_SHORT·(d_max_eff - x)
         t ≥ C_HOLD ·(x - d_min_eff)
         x, t ≥ 0
    d_max_eff and d_min_eff are Γ-interpolated between mean and extreme.
    """
    d_min = float(np.min(scenarios))
    d_max = float(np.max(scenarios))
    d_mean = float(np.mean(scenarios))

    # Γ-interpolation between mean and extreme (Bertsimas-Sim spirit)
    d_min_eff = d_mean - gamma * (d_mean - d_min)
    d_max_eff = d_mean + gamma * (d_max - d_mean)

    x = cp.Variable(nonneg=True)
    t = cp.Variable(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(t),
        [t >= C_SHORT * (d_max_eff - x),
         t >= C_HOLD  * (x - d_min_eff)]
    )
    prob.solve(solver=SOLVER, verbose=False)
    return float(x.value) if x.value is not None else np.nan


def solve_dro(scenarios: np.ndarray, gamma2: float = 1.0) -> float:
    """
    Distributionally Robust Optimisation — moment-based (Scarf 1958).
    Ambiguity set: distributions consistent with estimated mean ± γ₂·std.
    Closed-form minimax solution (Scarf formula):

        x* = μ_eff + (σ_eff / 2) · (√(b/h) - √(h/b))

    where μ_eff, σ_eff are the ambiguity-set moments.
    γ₂ = 0 → point estimate; γ₂ = 1 → full moment uncertainty.
    """
    mu_hat    = float(np.mean(scenarios))
    sigma_hat = float(np.std(scenarios, ddof=1))

    # Moment ambiguity: expand std by γ₂ factor
    sigma_eff = sigma_hat * (1 + gamma2)
    mu_eff    = mu_hat

    b, h = C_SHORT, C_HOLD
    x_star = mu_eff + (sigma_eff / 2.0) * (np.sqrt(b / h) - np.sqrt(h / b))
    return float(max(x_star, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(x: float, demands: np.ndarray) -> dict:
    """Compute out-of-sample newsvendor cost statistics for order quantity x."""
    shortage = np.maximum(demands - x, 0.0)
    surplus  = np.maximum(x - demands, 0.0)
    costs    = C_SHORT * shortage + C_HOLD * surplus
    return {
        "order_qty" : x,
        "mean_cost" : float(np.mean(costs)),
        "std_cost"  : float(np.std(costs)),
        "p95_cost"  : float(np.percentile(costs, 95)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — MAIN COMPARISON ACROSS SAMPLE SIZES
# ─────────────────────────────────────────────────────────────────────────────

def run_main_comparison():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — SP / RO / DRO Comparison Across Sample Sizes")
    print("=" * 70)
    print(f"\n  Oracle x* = {ORACLE:.2f}  (true LogNormal, CR = {CR:.3f})")
    print(f"  Out-of-sample evaluation: n = {N_TEST} draws (seed={TEST_SEED})\n")

    header = f"  {'n':>6}  {'Method':>6}  {'x*':>8}  {'Δ oracle':>9}  "
    header += f"{'Mean cost':>10}  {'Std cost':>10}  {'P95 cost':>10}"
    print(header)
    print("  " + "-" * 68)

    oracle_eval = evaluate(ORACLE, TEST_DEMANDS)
    records = []

    for n in SAMPLE_SIZES:
        scenarios = sample_demand(n, seed=TRAIN_SEED)

        for method, x in [
            ("SP",  solve_sp(scenarios)),
            ("RO",  solve_ro(scenarios, gamma=1.0)),
            ("DRO", solve_dro(scenarios, gamma2=1.0)),
        ]:
            ev = evaluate(x, TEST_DEMANDS)
            delta = x - ORACLE
            print(f"  {n:>6}  {method:>6}  {x:>8.2f}  {delta:>+9.2f}  "
                  f"{ev['mean_cost']:>10.2f}  {ev['std_cost']:>10.2f}  "
                  f"{ev['p95_cost']:>10.2f}")
            records.append({"n": n, "method": method, **ev, "delta_oracle": delta})

        # Oracle row (once per n block for reference)
        print(f"  {n:>6}  {'Oracle':>6}  {ORACLE:>8.2f}  {'0.00':>9}  "
              f"{oracle_eval['mean_cost']:>10.2f}  {oracle_eval['std_cost']:>10.2f}  "
              f"{oracle_eval['p95_cost']:>10.2f}")
        print("  " + "-" * 68)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — RO GAMMA SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_ro_gamma_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — RO Robustness Budget Γ Sweep  (n=50)")
    print("=" * 70)
    print(f"\n  Γ = 0 → nominal mean solution;  Γ = 1 → full box-robust solution.")
    print(f"  {'Γ':>6}  {'x*':>8}  {'Δ oracle':>9}  "
          f"{'Mean cost':>10}  {'P95 cost':>10}")
    print("  " + "-" * 50)

    gammas = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]
    scenarios = sample_demand(50, seed=TRAIN_SEED)
    records = []

    for g in gammas:
        x  = solve_ro(scenarios, gamma=g)
        ev = evaluate(x, TEST_DEMANDS)
        delta = x - ORACLE
        print(f"  {g:>6.2f}  {x:>8.2f}  {delta:>+9.2f}  "
              f"{ev['mean_cost']:>10.2f}  {ev['p95_cost']:>10.2f}")
        records.append({"gamma": g, **ev, "delta_oracle": delta})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3 — DRO GAMMA2 SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_dro_gamma2_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — DRO Moment Ambiguity γ₂ Sweep  (n=50)")
    print("=" * 70)
    print(f"\n  γ₂ = 0 → point-estimate moments;  γ₂ > 0 → expanded std ambiguity.")
    print(f"  {'γ₂':>6}  {'x*':>8}  {'Δ oracle':>9}  "
          f"{'Mean cost':>10}  {'P95 cost':>10}")
    print("  " + "-" * 50)

    gamma2s = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
    scenarios = sample_demand(50, seed=TRAIN_SEED)
    records = []

    for g2 in gamma2s:
        x  = solve_dro(scenarios, gamma2=g2)
        ev = evaluate(x, TEST_DEMANDS)
        delta = x - ORACLE
        print(f"  {g2:>6.2f}  {x:>8.2f}  {delta:>+9.2f}  "
              f"{ev['mean_cost']:>10.2f}  {ev['p95_cost']:>10.2f}")
        records.append({"gamma2": g2, **ev, "delta_oracle": delta})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4 — SP DISTRIBUTIONAL SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def run_sp_distributional_sensitivity():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 — SP Sensitivity to Distributional Misspecification")
    print("=" * 70)
    print("""
  True demand: LogNormal(log_mu=4.8, log_sigma=0.20)
  Scenario sets are drawn from four different assumed distributions,
  all calibrated to the same empirical mean and variance.
  Shows how SP solution quality degrades under wrong distributional assumptions.
""")
    print(f"  {'Distribution':>22}  {'x*':>8}  {'Δ oracle':>9}  "
          f"{'Mean cost':>10}  {'P95 cost':>10}")
    print("  " + "-" * 62)

    n = 50
    rng = np.random.default_rng(TRAIN_SEED)

    # Reference moments from true distribution
    true_mean  = np.exp(LOG_MU + 0.5 * LOG_SIGMA**2)
    true_var   = (np.exp(LOG_SIGMA**2) - 1) * np.exp(2 * LOG_MU + LOG_SIGMA**2)
    true_std   = np.sqrt(true_var)

    records = []

    # 1. Correct: LogNormal
    sc_lognorm = rng.lognormal(mean=LOG_MU, sigma=LOG_SIGMA, size=n)
    x = solve_sp(sc_lognorm)
    ev = evaluate(x, TEST_DEMANDS)
    print(f"  {'LogNormal (correct)':>22}  {x:>8.2f}  {x-ORACLE:>+9.2f}  "
          f"{ev['mean_cost']:>10.2f}  {ev['p95_cost']:>10.2f}")
    records.append({"dist": "LogNormal (correct)", **ev, "delta_oracle": x - ORACLE})

    # 2. Normal (same mean, std)
    sc_normal = np.maximum(rng.normal(loc=true_mean, scale=true_std, size=n), 0.1)
    x = solve_sp(sc_normal)
    ev = evaluate(x, TEST_DEMANDS)
    print(f"  {'Normal':>22}  {x:>8.2f}  {x-ORACLE:>+9.2f}  "
          f"{ev['mean_cost']:>10.2f}  {ev['p95_cost']:>10.2f}")
    records.append({"dist": "Normal", **ev, "delta_oracle": x - ORACLE})

    # 3. Uniform [mean-2σ, mean+2σ]
    lo = max(true_mean - 2 * true_std, 0.1)
    hi = true_mean + 2 * true_std
    sc_uniform = rng.uniform(lo, hi, size=n)
    x = solve_sp(sc_uniform)
    ev = evaluate(x, TEST_DEMANDS)
    print(f"  {'Uniform':>22}  {x:>8.2f}  {x-ORACLE:>+9.2f}  "
          f"{ev['mean_cost']:>10.2f}  {ev['p95_cost']:>10.2f}")
    records.append({"dist": "Uniform", **ev, "delta_oracle": x - ORACLE})

    # 4. Exponential (same mean, heavier tail)
    sc_exp = rng.exponential(scale=true_mean, size=n)
    x = solve_sp(sc_exp)
    ev = evaluate(x, TEST_DEMANDS)
    print(f"  {'Exponential':>22}  {x:>8.2f}  {x-ORACLE:>+9.2f}  "
          f"{ev['mean_cost']:>10.2f}  {ev['p95_cost']:>10.2f}")
    records.append({"dist": "Exponential", **ev, "delta_oracle": x - ORACLE})

    print(f"\n  Oracle: {ORACLE:.2f}")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(main_df, ro_df, dro_df, dist_df):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel 1: Order quantity vs n ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for method, marker in [("SP", "o"), ("RO", "s"), ("DRO", "^")]:
        sub = main_df[main_df["method"] == method]
        ax1.plot(sub["n"], sub["order_qty"], marker=marker, label=method)
    ax1.axhline(ORACLE, color="black", linestyle="--", linewidth=1.2, label="Oracle")
    ax1.set_xscale("log")
    ax1.set_xlabel("Sample size n")
    ax1.set_ylabel("Order quantity x*")
    ax1.set_title("Exp 1 — Order Quantity vs Sample Size")
    ax1.legend()

    # ── Panel 2: Mean cost vs n ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    oracle_cost = evaluate(ORACLE, TEST_DEMANDS)["mean_cost"]
    for method, marker in [("SP", "o"), ("RO", "s"), ("DRO", "^")]:
        sub = main_df[main_df["method"] == method]
        ax2.plot(sub["n"], sub["mean_cost"], marker=marker, label=method)
    ax2.axhline(oracle_cost, color="black", linestyle="--", linewidth=1.2, label="Oracle")
    ax2.set_xscale("log")
    ax2.set_xlabel("Sample size n")
    ax2.set_ylabel("Expected cost")
    ax2.set_title("Exp 1 — Expected Cost vs Sample Size")
    ax2.legend()

    # ── Panel 3: RO Gamma sweep ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ro_df["gamma"], ro_df["order_qty"], "s-", color="tab:orange", label="RO x*")
    ax3.axhline(ORACLE, color="black", linestyle="--", linewidth=1.2, label="Oracle")
    ax3b = ax3.twinx()
    ax3b.plot(ro_df["gamma"], ro_df["mean_cost"], "s--", color="tab:red", alpha=0.6, label="Mean cost")
    ax3b.set_ylabel("Expected cost", color="tab:red")
    ax3.set_xlabel("Robustness budget Γ")
    ax3.set_ylabel("Order quantity x*")
    ax3.set_title("Exp 2 — RO Γ Sweep (n=50)")
    ax3.legend(loc="upper left")

    # ── Panel 4: DRO gamma2 sweep ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(dro_df["gamma2"], dro_df["order_qty"], "^-", color="tab:green", label="DRO x*")
    ax4.axhline(ORACLE, color="black", linestyle="--", linewidth=1.2, label="Oracle")
    ax4b = ax4.twinx()
    ax4b.plot(dro_df["gamma2"], dro_df["mean_cost"], "^--", color="tab:red", alpha=0.6, label="Mean cost")
    ax4b.set_ylabel("Expected cost", color="tab:red")
    ax4.set_xlabel("Moment ambiguity γ₂")
    ax4.set_ylabel("Order quantity x*")
    ax4.set_title("Exp 3 — DRO γ₂ Sweep (n=50)")
    ax4.legend(loc="upper left")

    plt.savefig("section35_figures.png", dpi=150, bbox_inches="tight")
    print("\n  [✓] Figures saved → section35_figures.png")
    plt.close()





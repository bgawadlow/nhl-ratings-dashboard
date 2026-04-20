"""
NHL Ratings Model — Private module (not tracked in git).
Contains proprietary model constants, projection logic, and financial calculations.
"""

import numpy as np
import pandas as pd

# ── Model Constants ─────────────────────────────────────────────────────────
# Contract model: Market Value = Replacement + ($/SPAR × pSPAR)
# Replacement anchored to league minimum ($0.775M); slope re-fit on 5,638 obs
# of historical contract data (see Contract Model Eval.R). Removes ~$0.40M
# of bottom-of-roster overvaluation vs unconstrained regression.
# Calibrated to the current salary cap (BASE_CAP_M); future seasons scale by cap growth.
CONTRACT_MODEL = {
    "F": {
        "replacement_m":       0.775,      # $M replacement-level value (= league min)
        "dollars_per_spar_m":  1.3572,     # $M per unit of pSPAR
        "dollars_per_win_m":   3.0537,     # $/SPAR × 2.25 (reference)
    },
    "D": {
        "replacement_m":       0.775,
        "dollars_per_spar_m":  1.4868,
        "dollars_per_win_m":   3.3453,
    },
}
# Salary cap used to calibrate the contract model ($M). Future seasons scale by Proj_Cap_M / BASE_CAP_M.
BASE_CAP_M = 95.5

OVR_PARAMS = {"F": (77.5, 1.5), "D": (77.5, 1.85)}

DRAFT_MAX_LOOKUP = dict(zip(
    range(1981, 2026),
    [211, 252, 242, 250, 252, 252, 252, 252, 252, 250,
     264, 264, 286, 286, 234, 241, 246, 258, 272, 293,
     289, 291, 292, 291, 230, 213, 211, 211, 211, 210,
     211, 211, 211, 210, 211, 211, 217, 217, 217, 216,
     223, 225, 224, 225, 224],
))

PERF_VARS = [
    "pEVO_SPAR", "pEVD_SPAR", "pPPO_SPAR", "pSHD_SPAR",
    "pTAKE_SPAR", "pDRAW_SPAR",
    "predict_all_toi", "predict_pp_toi", "predict_sh_toi",
    "Hits82", "Blk82", "PTS82", "G82", "EV_PTS82", "EV_G82",
]

SPAR_COMPONENTS = [
    "pEVO_SPAR", "pEVD_SPAR", "pPPO_SPAR", "pSHD_SPAR",
    "pTAKE_SPAR", "pDRAW_SPAR",
]

# Known salary caps by season start year
KNOWN_CAPS = {2026: 95.5, 2027: 104.0, 2028: 113.5}

# Fixed value for all undrafted players (instead of per-year max_pick + 1)
UNDRAFTED_PICK_VALUE = 250

# League minimum salary ($M)
LEAGUE_MIN_SALARY = 0.775


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_cap(season_start_year):
    """Return projected salary cap ($M) for a given season start year."""
    if season_start_year in KNOWN_CAPS:
        return KNOWN_CAPS[season_start_year]
    return round(113.5 * (1.05 ** (season_start_year - 2028)), 1)


def ovr_f(spar_val):
    return OVR_PARAMS["F"][0] + OVR_PARAMS["F"][1] * spar_val


def ovr_d(spar_val):
    return OVR_PARAMS["D"][0] + OVR_PARAMS["D"][1] * spar_val


def compute_ovr(pos, spar_val):
    o_base, o_mult = OVR_PARAMS[pos]
    return round(o_base + o_mult * spar_val)


# ── Similarity feature weights (skill scheme) ──────────────────────────────
# Backtest (800 player-seasons, 3-year horizon) showed this weighting:
#   - Cuts overall MAE by 11.5% vs equal-weight baseline
#   - Cuts top-tier (pSPAR 6+) RMSE by 47% (3.02 -> 1.71)
#   - Fixes Bouchard-Kovacevic-style nonsensical matches by making primary
#     SPAR components (EVO/EVD/PPO) dominate the similarity calculation.
# Base weights apply to current-season features; delta/slope weights apply to
# trajectory features (d_*, slope2_*) which are noisier.
_FEATURE_WEIGHTS = {
    # (base_weight, delta_weight)
    "pEVO_SPAR":         (3.0, 0.75),
    "pEVD_SPAR":         (3.0, 0.75),
    "pPPO_SPAR":         (3.0, 0.75),
    "pSHD_SPAR":         (2.0, 0.50),
    "pTAKE_SPAR":        (1.0, 0.25),
    "pDRAW_SPAR":        (1.0, 0.25),
    "predict_all_toi":   (2.0, 0.50),
    "predict_pp_toi":    (2.0, 0.50),
    "predict_sh_toi":    (2.0, 0.50),
    "PTS82":             (1.0, 0.25),
    "G82":               (1.0, 0.25),
    "EV_PTS82":          (1.0, 0.25),
    "EV_G82":            (1.0, 0.25),
    "Hits82":            (1.0, 0.25),
    "Blk82":             (1.0, 0.25),
    # Contextual / draft features
    "is_undrafted":      (0.5, 0.0),
    "Draft_Ov_Log":      (0.5, 0.0),
    # NHL experience — separates rookies from veterans at same age
    "NHL_Seasons":       (2.0, 0.0),
}
_DEFAULT_WEIGHT = (0.5, 0.25)


def skill_weight_mask(z_vars):
    """
    Build a per-feature weight vector for comp similarity (skill scheme).
    Expects feature names of the form 'z_<base>', 'z_d_<base>', or 'z_slope2_<base>'.
    Trajectory features (d_ and slope2_) get the lower delta-weight.
    """
    weights = np.empty(len(z_vars), dtype=float)
    for i, zv in enumerate(z_vars):
        name = zv[2:] if zv.startswith("z_") else zv
        is_delta = name.startswith("d_") or name.startswith("slope2_")
        base = name[2:] if name.startswith("d_") else (name[7:] if name.startswith("slope2_") else name)
        base_w, delta_w = _FEATURE_WEIGHTS.get(base, _DEFAULT_WEIGHT)
        weights[i] = delta_w if is_delta else base_w
    return weights


# ── Dataset Builder ─────────────────────────────────────────────────────────

def build_scaled_dataset(spar_df, draft_df):
    """Merge SPAR + draft info, compute deltas, slopes, per-60 rates & z-scores."""
    df = spar_df.copy()

    # ── Merge draft info ──
    if "Player_Key" in draft_df.columns:
        draft_map = draft_df.drop_duplicates(subset="Player_Key", keep="first")
        df = df.merge(
            draft_map[["Player_Key", "Draft Yr", "Draft Ov"]],
            left_on="Player", right_on="Player_Key", how="left",
        ).drop(columns=["Player_Key"])
    else:
        draft_map = draft_df.drop_duplicates(subset="Player", keep="first")
        df = df.merge(
            draft_map[["Player", "Draft Yr", "Draft Ov"]],
            on="Player", how="left",
        )

    # ── Undrafted handling (improvement 1) ──
    # All undrafted players get a single fixed value instead of per-year max_pick+1
    df["is_undrafted"] = df["Draft Ov"].isna().astype(float)
    df["Draft_Ov_Clean"] = np.where(
        df["Draft Ov"].notna(),
        df["Draft Ov"],
        UNDRAFTED_PICK_VALUE,
    )
    df["Draft_Ov_Log"] = np.log(df["Draft_Ov_Clean"].astype(float))

    df["Season_Num"] = df["Season"].str[3:5].astype(int) + 2000

    for col in PERF_VARS + ["pSPAR"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── Per-60 rates (kept for reference / v1 compatibility) ──
    df["ev_toi"] = (
        df["predict_all_toi"]
        - df["predict_pp_toi"].fillna(0)
        - df["predict_sh_toi"].fillna(0)
    )
    df["rate_EVO"] = np.where(df["ev_toi"] > 0, df["pEVO_SPAR"] * 60 / (df["ev_toi"] * 82), 0)
    df["rate_EVD"] = np.where(df["ev_toi"] > 0, df["pEVD_SPAR"] * 60 / (df["ev_toi"] * 82), 0)
    df["rate_PPO"] = np.where(df["predict_pp_toi"] > 0, df["pPPO_SPAR"] * 60 / (df["predict_pp_toi"] * 82), 0)
    df["rate_SHD"] = np.where(df["predict_sh_toi"] > 0, df["pSHD_SPAR"] * 60 / (df["predict_sh_toi"] * 82), 0)
    df["rate_TAKE"] = np.where(df["predict_all_toi"] > 0, df["pTAKE_SPAR"] * 60 / (df["predict_all_toi"] * 82), 0)
    df["rate_DRAW"] = np.where(df["predict_all_toi"] > 0, df["pDRAW_SPAR"] * 60 / (df["predict_all_toi"] * 82), 0)

    # ── Deltas (improvement 3a): NaN for first-season instead of 0 ──
    df = df.sort_values(["Player", "Season_Num"])
    for var in PERF_VARS:
        if var in df.columns:
            df[f"d_{var}"] = df.groupby("Player")[var].diff()  # NaN for first season

    # ── NHL experience: seasons played to-date (including current) ──
    # Differentiates rookies from veterans at the same age (e.g., 25yo late-
    # bloomer on year 1 vs 25yo top pick on year 5+). Avoids cross-tenure
    # comp matches that the age filter alone can't prevent.
    df["NHL_Seasons"] = df.groupby("Player").cumcount() + 1

    # Lagged features were tested (see backtest_nhl_seasons.py) and rejected:
    # they helped mid-tier (pSPAR 3-6) RMSE by ~1% but hurt top-tier (6+) by 4%
    # because stars have stable trajectories and lags add noise without
    # discrimination. Deltas + slopes already capture trajectory shape.

    # ── 2-year slope / trend (improvement 3b) — vectorized ──
    # Uses groupby.shift at the DataFrame level (fast) rather than a
    # per-group Python transform (30x slower).
    g_player = df.groupby("Player")
    for var in PERF_VARS:
        if var in df.columns:
            prev1 = g_player[var].shift(1)
            prev2 = g_player[var].shift(2)
            slope3 = (df[var] - prev2) / 2.0
            slope2 = df[var] - prev1
            df[f"slope2_{var}"] = slope3.where(prev2.notna(), slope2.where(prev1.notna(), np.nan))

    # ── Build variable lists for z-scoring ──
    delta_vars = [f"d_{v}" for v in PERF_VARS if v in df.columns]
    slope_vars = [f"slope2_{v}" for v in PERF_VARS if v in df.columns]
    all_vars = (
        [v for v in PERF_VARS if v in df.columns]
        + delta_vars
        + slope_vars
        + ["Draft_Ov_Log", "is_undrafted", "NHL_Seasons"]
    )

    # Vectorized z-scoring within Season (replaces per-group Python transform)
    g_season = df.groupby("Season")
    for var in all_vars:
        if var in df.columns:
            mean_ = g_season[var].transform("mean")
            std_  = g_season[var].transform("std")
            # Avoid div-by-zero: fill missing / zero std with 0 to match old safe_scale behavior
            z = (df[var] - mean_) / std_.replace(0, np.nan)
            df[f"z_{var}"] = z.fillna(0)

    return df.sort_values(["Player", "Age"])


# ── Survival Table ──────────────────────────────────────────────────────────

def build_survival_table(df):
    """P(still playing next season | position, age, SPAR bucket)."""
    df = df.copy()
    df["spar_bucket"] = pd.cut(
        df["pSPAR"], bins=[-np.inf, 0, 3, 7, 12, np.inf],
        labels=["below_repl", "low", "mid", "high", "elite"],
    )
    df = df.sort_values(["Player", "Age"])
    df["has_next"] = df.groupby("Player")["Age"].shift(-1) == df["Age"] + 1
    df["has_next"] = df["has_next"].fillna(False).astype(int)

    surv = (
        df.groupby(["Position", "Age", "spar_bucket"])
        .agg(n=("has_next", "count"), survived=("has_next", "sum"))
        .reset_index()
    )
    surv["rate"] = np.where(surv["n"] >= 10, surv["survived"] / surv["n"], np.nan)

    pos_age = (
        df.groupby(["Position", "Age"])["has_next"]
        .mean().reset_index(name="rate_avg")
    )
    surv = surv.merge(pos_age, on=["Position", "Age"], how="left")
    surv["rate"] = surv["rate"].fillna(surv["rate_avg"]).fillna(0.5)
    return surv


def get_survival_prob(pos, age, spar_val, surv_tbl):
    bucket = pd.cut(
        [spar_val], bins=[-np.inf, 0, 3, 7, 12, np.inf],
        labels=["below_repl", "low", "mid", "high", "elite"],
    )[0]
    match = surv_tbl[
        (surv_tbl["Position"] == pos)
        & (surv_tbl["Age"] == age)
        & (surv_tbl["spar_bucket"] == bucket)
    ]
    return float(match["rate"].iloc[0]) if not match.empty else 0.7


# ── Logistic Survival Model (improvement 5) ──────────────────────────────

def _sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def build_survival_model(df):
    """Fit logistic regression for P(survive next season) via hand-rolled IRLS.

    Features: Age, pSPAR, Age*pSPAR interaction, is_D (position).
    Returns dict with weights, intercept, and standardization params.
    """
    df = df.copy()
    df = df.sort_values(["Player", "Age"])
    df["has_next"] = df.groupby("Player")["Age"].shift(-1) == df["Age"] + 1
    df["has_next"] = df["has_next"].fillna(False).astype(float)

    # Drop rows with missing key columns
    model_df = df[["Age", "pSPAR", "Position", "has_next"]].dropna().copy()
    model_df["is_D"] = (model_df["Position"] == "D").astype(float)
    model_df["age_spar"] = model_df["Age"] * model_df["pSPAR"]

    # Standardize continuous features
    stats = {}
    for col in ["Age", "pSPAR", "age_spar"]:
        mu, sd = model_df[col].mean(), model_df[col].std()
        if sd < 1e-9:
            sd = 1.0
        stats[col] = (mu, sd)
        model_df[f"{col}_z"] = (model_df[col] - mu) / sd

    # Build design matrix  [age_z, spar_z, interaction_z, is_D]
    X = model_df[["Age_z", "pSPAR_z", "age_spar_z", "is_D"]].values
    y = model_df["has_next"].values
    n, k = X.shape

    # IRLS with L2 regularization (lambda=0.01 for numerical stability)
    w = np.zeros(k)
    b = 0.0
    lam = 0.01

    for _ in range(30):
        z = X @ w + b
        p = _sigmoid(z)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        r = p * (1 - p)
        grad_w = X.T @ (p - y) / n + lam * w
        grad_b = np.mean(p - y)
        H = (X.T * r) @ X / n + lam * np.eye(k)
        try:
            dw = np.linalg.solve(H, grad_w)
        except np.linalg.LinAlgError:
            dw = grad_w * 0.01
        w -= dw
        b -= 0.1 * grad_b  # smaller step for intercept

    return {
        "weights": w,
        "intercept": b,
        "stats": stats,  # {col: (mean, std)}
        "features": ["Age", "pSPAR", "age_spar", "is_D"],
    }


def get_survival_prob_v3(pos, age, spar_val, survival_model):
    """Predict P(survive next season) using the logistic model."""
    stats = survival_model["stats"]
    age_z = (age - stats["Age"][0]) / stats["Age"][1]
    spar_z = (spar_val - stats["pSPAR"][0]) / stats["pSPAR"][1]
    interaction_z = (age * spar_val - stats["age_spar"][0]) / stats["age_spar"][1]
    is_d = 1.0 if pos == "D" else 0.0

    x = np.array([age_z, spar_z, interaction_z, is_d])
    z = np.dot(survival_model["weights"], x) + survival_model["intercept"]
    return float(np.clip(_sigmoid(z), 0.01, 0.99))


# ── v1 Projection ──────────────────────────────────────────────────────────

def get_projection(target_name, target_season, dataset):
    """Run similarity-based projection for a player."""
    target = dataset[(dataset["Player"] == target_name) & (dataset["Season"] == target_season)]
    if target.empty:
        return None, None, None
    target = target.iloc[0]
    t_age = target["Age"]
    t_pos = target["Position"]

    delta_vars = [f"d_{v}" for v in PERF_VARS if v in dataset.columns]
    active_vars = (
        [v for v in PERF_VARS if v in dataset.columns]
        + delta_vars
        + (["Draft_Ov_Log"] if t_age <= 26 else [])
    )
    z_vars = [f"z_{v}" for v in active_vars if f"z_{v}" in dataset.columns]

    target_stats = target[z_vars].values.astype(float)

    cohort = dataset[
        (dataset["Position"] == t_pos)
        & (dataset["Age"] == t_age)
        & (dataset["Player"] != target_name)
    ].copy()
    if cohort.empty:
        return None, None, None

    cohort_stats = cohort[z_vars].values.astype(float)
    dists = np.sqrt(np.nansum((cohort_stats - target_stats) ** 2, axis=1))
    cohort["weight"] = 1 / (dists ** 2 + 0.05)
    cohort = cohort.sort_values("weight", ascending=False)

    projections_list = []
    current_spar = float(target["pSPAR"])
    for i in range(1, 9):
        next_age = t_age + i
        delta_data = dataset[dataset["Player"].isin(cohort["Player"])]
        pairs = delta_data[delta_data["Age"].isin([next_age - 1, next_age])]
        pairs = pairs.groupby("Player").filter(lambda g: len(g) == 2)
        if not pairs.empty:
            yoy = (
                pairs.sort_values(["Player", "Age"])
                .groupby("Player")["pSPAR"]
                .apply(lambda x: x.iloc[1] - x.iloc[0])
                .reset_index(name="yoy")
            )
            yoy = yoy.merge(cohort[["Player", "weight"]], on="Player")
            avg_change = np.average(yoy["yoy"], weights=yoy["weight"])
        else:
            avg_change = -0.5
        current_spar += avg_change
        projections_list.append({
            "Age": next_age,
            "Predicted_pSPAR": current_spar,
            "Season_Index": f"+{i}",
        })

    o_base, o_mult = OVR_PARAMS[t_pos]
    contract = CONTRACT_MODEL[t_pos]

    proj_df = pd.DataFrame(projections_list)
    proj_df = pd.concat([
        pd.DataFrame([{
            "Age": t_age,
            "Predicted_pSPAR": float(target["pSPAR"]),
            "Season_Index": "Current",
        }]),
        proj_df,
    ]).reset_index(drop=True)

    proj_df["Predicted_OVR"] = (o_base + o_mult * proj_df["Predicted_pSPAR"]).round(0).astype(int)

    base_cap_year = int(target["Season_Num"]) + 1
    proj_df["Proj_Cap_M"] = [get_cap(base_cap_year + i) for i in range(len(proj_df))]

    proj_df["Season_Label"] = [
        f"{(base_cap_year + i) % 100 - 1:02d}-{(base_cap_year + i) % 100:02d}"
        for i in range(len(proj_df))
    ]

    base_value = contract["replacement_m"] + contract["dollars_per_spar_m"] * proj_df["Predicted_pSPAR"]
    proj_df["Market_Value_M"] = base_value * (proj_df["Proj_Cap_M"] / BASE_CAP_M)

    future = proj_df[proj_df["Season_Index"] != "Current"].reset_index(drop=True)
    contract_table = pd.DataFrame({
        "Term": range(1, 9),
        "Total_Value_M": future["Market_Value_M"].cumsum().round(2),
    })
    contract_table["AAV_M"] = (contract_table["Total_Value_M"] / contract_table["Term"]).round(3)

    season_counts = dataset.groupby("Player")["Season"].nunique()
    experienced = season_counts[season_counts > 3].index
    comps_pool = cohort[cohort["Player"].isin(experienced)]
    top_comps = comps_pool.head(10)[["Player", "Season", "Age", "pSPAR", "predict_all_toi", "weight"]].copy()
    top_comps.columns = ["Player", "Season", "Age", "pSPAR", "TOI", "Similarity"]

    return proj_df, contract_table, top_comps


# ── v2 Projection (component-level aging, decay, TOI-split, survival) ──────

def get_projection_v2(target_name, target_season, dataset, survival_table,
                      decay_rate=0.85, era_decay=0.92):
    """v2: component-level aging, similarity decay, TOI-split, survival bias."""
    target = dataset[(dataset["Player"] == target_name) & (dataset["Season"] == target_season)]
    if target.empty:
        return None, None, None
    target = target.iloc[0]
    t_age = int(target["Age"])
    t_pos = target["Position"]

    delta_vars_v = [f"d_{v}" for v in PERF_VARS if v in dataset.columns]
    active_vars = (
        [v for v in PERF_VARS if v in dataset.columns]
        + delta_vars_v
        + (["Draft_Ov_Log"] if t_age <= 26 else [])
    )
    z_vars = [f"z_{v}" for v in active_vars if f"z_{v}" in dataset.columns]
    target_stats = target[z_vars].values.astype(float)

    cohort = dataset[
        (dataset["Position"] == t_pos)
        & (dataset["Age"] == t_age)
        & (dataset["Player"] != target_name)
    ].copy()
    if cohort.empty:
        return None, None, None

    cohort_stats = cohort[z_vars].values.astype(float)
    dists = np.sqrt(np.nansum((cohort_stats - target_stats) ** 2, axis=1))
    cohort["base_weight"] = 1 / (dists ** 2 + 0.05)
    cohort = cohort.sort_values("base_weight", ascending=False)

    rate_keys = ["rate_EVO", "rate_EVD", "rate_PPO", "rate_SHD", "rate_TAKE", "rate_DRAW"]
    curr_rates = {k: float(target[k]) for k in rate_keys}
    curr_toi = {
        "ev": float(target["ev_toi"]),
        "pp": float(target.get("predict_pp_toi", 0) or 0),
        "sh": float(target.get("predict_sh_toi", 0) or 0),
    }

    projections_list = []
    for i in range(1, 9):
        next_age = t_age + i

        comp_players = cohort["Player"].values
        delta_pool = dataset[dataset["Player"].isin(comp_players)]
        pairs = delta_pool[delta_pool["Age"].isin([next_age - 1, next_age])]
        pairs = pairs.groupby("Player").filter(lambda g: len(g) == 2)

        if not pairs.empty:
            pairs_sorted = pairs.sort_values(["Player", "Age"])
            prev = pairs_sorted.groupby("Player").nth(0).reset_index()
            curr = pairs_sorted.groupby("Player").nth(1).reset_index()
            yoy = pd.DataFrame({"Player": prev["Player"]})
            for rk in rate_keys:
                yoy[f"d_{rk}"] = curr[rk].values - prev[rk].values
            yoy["d_ev_toi"] = curr["ev_toi"].values - prev["ev_toi"].values
            yoy["d_pp_toi"] = curr["predict_pp_toi"].fillna(0).values - prev["predict_pp_toi"].fillna(0).values
            yoy["d_sh_toi"] = curr["predict_sh_toi"].fillna(0).values - prev["predict_sh_toi"].fillna(0).values

            yoy = yoy.merge(cohort[["Player", "base_weight"]], on="Player")
            max_season = prev["Season_Num"].max()
            yoy["era_weight"] = era_decay ** (max_season - prev["Season_Num"].values)
            yoy["weight"] = yoy["base_weight"] * (decay_rate ** i) * yoy["era_weight"]

            w = yoy["weight"].values
            avg_d = {col: np.average(yoy[col], weights=w) for col in yoy.columns if col.startswith("d_")}
        else:
            avg_d = {f"d_{rk}": -0.001 for rk in rate_keys}
            avg_d["d_ev_toi"] = -0.3
            avg_d["d_pp_toi"] = -0.05
            avg_d["d_sh_toi"] = -0.02

        for rk in rate_keys:
            curr_rates[rk] += avg_d.get(f"d_{rk}", 0)

        curr_toi["ev"] = max(curr_toi["ev"] + avg_d.get("d_ev_toi", 0), 0)
        curr_toi["pp"] = max(curr_toi["pp"] + avg_d.get("d_pp_toi", 0), 0)
        curr_toi["sh"] = max(curr_toi["sh"] + avg_d.get("d_sh_toi", 0), 0)
        all_toi = curr_toi["ev"] + curr_toi["pp"] + curr_toi["sh"]

        comp_vals = {
            "pEVO": (curr_rates["rate_EVO"] / 60) * curr_toi["ev"] * 82,
            "pEVD": (curr_rates["rate_EVD"] / 60) * curr_toi["ev"] * 82,
            "pPPO": (curr_rates["rate_PPO"] / 60) * curr_toi["pp"] * 82,
            "pSHD": (curr_rates["rate_SHD"] / 60) * curr_toi["sh"] * 82,
            "pTAKE": (curr_rates["rate_TAKE"] / 60) * all_toi * 82,
            "pDRAW": (curr_rates["rate_DRAW"] / 60) * all_toi * 82,
        }
        raw_spar = sum(comp_vals.values())

        surv_prob = get_survival_prob(t_pos, next_age - 1, raw_spar, survival_table)

        projections_list.append({
            "Age": next_age,
            "Predicted_pSPAR": raw_spar,
            "Survival_Prob": surv_prob,
            "TOI": round(all_toi, 1),
            "Season_Index": f"+{i}",
        })

    o_base, o_mult = OVR_PARAMS[t_pos]
    contract = CONTRACT_MODEL[t_pos]

    proj_df = pd.DataFrame(projections_list)
    proj_df = pd.concat([
        pd.DataFrame([{
            "Age": t_age,
            "Predicted_pSPAR": float(target["pSPAR"]),
            "Survival_Prob": 1.0,
            "TOI": round(float(target["predict_all_toi"]), 1),
            "Season_Index": "Current",
        }]),
        proj_df,
    ]).reset_index(drop=True)

    proj_df["Predicted_OVR"] = (o_base + o_mult * proj_df["Predicted_pSPAR"]).round(0).astype(int)

    base_cap_year = int(target["Season_Num"]) + 1
    proj_df["Proj_Cap_M"] = [get_cap(base_cap_year + i) for i in range(len(proj_df))]

    proj_df["Season_Label"] = [
        f"{(base_cap_year + i) % 100 - 1:02d}-{(base_cap_year + i) % 100:02d}"
        for i in range(len(proj_df))
    ]

    base_value = contract["replacement_m"] + contract["dollars_per_spar_m"] * proj_df["Predicted_pSPAR"]
    proj_df["Market_Value_M"] = base_value * (proj_df["Proj_Cap_M"] / BASE_CAP_M)

    future = proj_df[proj_df["Season_Index"] != "Current"].reset_index(drop=True)
    contract_table = pd.DataFrame({
        "Term": range(1, 9),
        "Total_Value_M": future["Market_Value_M"].cumsum().round(2),
    })
    contract_table["AAV_M"] = (contract_table["Total_Value_M"] / contract_table["Term"]).round(3)

    season_counts = dataset.groupby("Player")["Season"].nunique()
    experienced = season_counts[season_counts > 3].index
    comps_pool = cohort[cohort["Player"].isin(experienced)]
    top_comps = comps_pool.head(10)[["Player", "Season", "Age", "pSPAR", "predict_all_toi", "base_weight"]].copy()
    top_comps.columns = ["Player", "Season", "Age", "pSPAR", "TOI", "Similarity"]

    return proj_df, contract_table, top_comps


# ── v3 Projection (all 7 improvements) ───────────────────────────────────

def get_projection_v3(target_name, target_season, dataset, survival_model,
                      decay_rate=0.85, era_decay=0.92):
    """v3: direct component aging, Gaussian kernel, logistic survival,
    improved undrafted/delta/slope features, era-anchored decay, market floor."""
    target = dataset[
        (dataset["Player"] == target_name) & (dataset["Season"] == target_season)
    ]
    if target.empty:
        return None, None, None
    target = target.iloc[0]
    t_age = int(target["Age"])
    t_pos = target["Position"]
    t_season_num = int(target["Season_Num"])

    # ── Build similarity feature list (improvements 1, 3) ──
    delta_vars_v = [f"d_{v}" for v in PERF_VARS if f"d_{v}" in dataset.columns]
    slope_vars_v = [f"slope2_{v}" for v in PERF_VARS if f"slope2_{v}" in dataset.columns]
    active_vars = (
        [v for v in PERF_VARS if v in dataset.columns]
        + delta_vars_v
        + slope_vars_v
        + ["is_undrafted", "NHL_Seasons"]
        + (["Draft_Ov_Log"] if t_age <= 26 else [])
    )
    z_vars = [f"z_{v}" for v in active_vars if f"z_{v}" in dataset.columns]

    # ── Skill-weighted feature mask (validated via backtest) ──
    # Primary SPAR components and TOI dominate; noisy trajectory features
    # are down-weighted. Eliminates nonsensical matches (e.g., high-PP
    # offensive D being matched to shutdown D with similar total TOI).
    weight_mask = skill_weight_mask(z_vars)

    target_stats = target[z_vars].values.astype(float) * weight_mask

    # ── Build cohort (same position, same age, different player) ──
    cohort = dataset[
        (dataset["Position"] == t_pos)
        & (dataset["Age"] == t_age)
        & (dataset["Player"] != target_name)
    ].copy()
    if cohort.empty:
        return None, None, None

    cohort_stats = cohort[z_vars].values.astype(float) * weight_mask

    # ── Gaussian kernel similarity (improvement 4) ──
    # Missing features (NaN) shouldn't penalize the distance — replace with 0.
    diffs = cohort_stats - target_stats
    diffs = np.where(np.isnan(diffs), 0.0, diffs)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    sigma = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    if sigma < 1e-6:
        sigma = 1.0
    cohort["base_weight"] = np.exp(-dists ** 2 / (2 * sigma ** 2))
    cohort = cohort.sort_values("base_weight", ascending=False)

    # ── Project each SPAR component directly (improvement 2) ──
    curr_components = {comp: float(target[comp]) for comp in SPAR_COMPONENTS}

    # Also track TOI as informational (not used to reconstruct SPAR)
    curr_toi = {
        "ev": float(target["ev_toi"]),
        "pp": float(target.get("predict_pp_toi", 0) or 0),
        "sh": float(target.get("predict_sh_toi", 0) or 0),
    }

    # ── Pre-build Player+Age -> pSPAR lookup once (outside horizon loop) ──
    # Avoids O(cohort * dataset) row scans for bounds calc on every year.
    pspar_lookup = dict(zip(
        zip(dataset["Player"].values, dataset["Age"].astype(int).values),
        dataset["pSPAR"].values,
    ))
    cohort_info = list(zip(
        cohort["Player"].values,
        cohort["Age"].astype(int).values,
        cohort["pSPAR"].values,
        cohort["base_weight"].values,
    ))

    projections_list = []
    for i in range(1, 9):
        next_age = t_age + i

        comp_players = cohort["Player"].values
        delta_pool = dataset[dataset["Player"].isin(comp_players)]
        pairs = delta_pool[delta_pool["Age"].isin([next_age - 1, next_age])]
        pairs = pairs.groupby("Player").filter(lambda g: len(g) == 2)

        if not pairs.empty:
            pairs_sorted = pairs.sort_values(["Player", "Age"])
            prev = pairs_sorted.groupby("Player").nth(0).reset_index()
            curr = pairs_sorted.groupby("Player").nth(1).reset_index()

            yoy = pd.DataFrame({"Player": prev["Player"]})

            # Component-level YoY deltas
            for comp in SPAR_COMPONENTS:
                yoy[f"d_{comp}"] = curr[comp].values - prev[comp].values

            # TOI deltas (informational)
            yoy["d_ev_toi"] = curr["ev_toi"].values - prev["ev_toi"].values
            yoy["d_pp_toi"] = (
                curr["predict_pp_toi"].fillna(0).values
                - prev["predict_pp_toi"].fillna(0).values
            )
            yoy["d_sh_toi"] = (
                curr["predict_sh_toi"].fillna(0).values
                - prev["predict_sh_toi"].fillna(0).values
            )

            yoy = yoy.merge(cohort[["Player", "base_weight"]], on="Player")

            # Era decay anchored to target season (improvement 6)
            yoy["era_weight"] = era_decay ** (t_season_num - prev["Season_Num"].values)
            yoy["weight"] = yoy["base_weight"] * (decay_rate ** i) * yoy["era_weight"]

            w = yoy["weight"].values
            avg_d = {
                col: np.average(yoy[col], weights=w)
                for col in yoy.columns if col.startswith("d_")
            }
        else:
            # Fallback when no comp transitions exist for this age
            avg_d = {f"d_{comp}": -0.05 for comp in SPAR_COMPONENTS}
            avg_d["d_ev_toi"] = -0.3
            avg_d["d_pp_toi"] = -0.05
            avg_d["d_sh_toi"] = -0.02

        # Apply component deltas directly
        for comp in SPAR_COMPONENTS:
            curr_components[comp] += avg_d.get(f"d_{comp}", 0)

        raw_spar = sum(curr_components.values())

        # Update TOI (informational only — does NOT feed into SPAR)
        curr_toi["ev"] = max(curr_toi["ev"] + avg_d.get("d_ev_toi", 0), 0)
        curr_toi["pp"] = max(curr_toi["pp"] + avg_d.get("d_pp_toi", 0), 0)
        curr_toi["sh"] = max(curr_toi["sh"] + avg_d.get("d_sh_toi", 0), 0)
        all_toi = curr_toi["ev"] + curr_toi["pp"] + curr_toi["sh"]

        # Logistic survival probability (improvement 5)
        surv_prob = get_survival_prob_v3(t_pos, next_age - 1, raw_spar, survival_model)

        # ── Prediction bounds: weighted P10/P90 of cumulative comp trajectories ──
        # For each comp, apply their (current -> current+i) pSPAR delta to the
        # target's current pSPAR. Uses O(1) dict lookups (pspar_lookup) built
        # once outside the horizon loop.
        target_pspar_now = float(target["pSPAR"])
        trajectories = []
        traj_weights  = []
        for c_name, c_base_age, c_pspar_now, c_weight in cohort_info:
            c_fut = pspar_lookup.get((c_name, c_base_age + i))
            if c_fut is None:
                continue
            c_delta = c_fut - c_pspar_now
            trajectories.append(target_pspar_now + c_delta)
            traj_weights.append(c_weight)

        if trajectories:
            vals = np.asarray(trajectories, dtype=float)
            wts  = np.asarray(traj_weights, dtype=float)
            order = np.argsort(vals)
            vals_s = vals[order]
            wts_s  = wts[order]
            cum = np.cumsum(wts_s) / wts_s.sum()
            pspar_p10 = float(np.interp(0.10, cum, vals_s))
            pspar_p90 = float(np.interp(0.90, cum, vals_s))
        else:
            pspar_p10 = float("nan")
            pspar_p90 = float("nan")

        projections_list.append({
            # pSPAR is a projection for the *next* season, so the displayed age
            # is incremented to match the season being projected.
            "Age": next_age + 1,
            "Predicted_pSPAR": raw_spar,
            "pSPAR_Low":  pspar_p10,
            "pSPAR_High": pspar_p90,
            "Survival_Prob": surv_prob,
            "TOI": round(all_toi, 1),
            "Season_Index": f"+{i}",
        })

    # ── Build output tables ──
    o_base, o_mult = OVR_PARAMS[t_pos]
    contract = CONTRACT_MODEL[t_pos]

    proj_df = pd.DataFrame(projections_list)
    proj_df = pd.concat([
        pd.DataFrame([{
            # target["pSPAR"] is next-season's projection → bump age to match.
            "Age": t_age + 1,
            "Predicted_pSPAR": float(target["pSPAR"]),
            "pSPAR_Low":  float(target["pSPAR"]),  # no uncertainty on current
            "pSPAR_High": float(target["pSPAR"]),
            "Survival_Prob": 1.0,
            "TOI": round(float(target["predict_all_toi"]), 1),
            "Season_Index": "Current",
        }]),
        proj_df,
    ]).reset_index(drop=True)

    proj_df["Predicted_OVR"] = (o_base + o_mult * proj_df["Predicted_pSPAR"]).round(0).astype(int)

    base_cap_year = int(target["Season_Num"]) + 1
    proj_df["Proj_Cap_M"] = [get_cap(base_cap_year + i) for i in range(len(proj_df))]
    proj_df["Season_Label"] = [
        f"{(base_cap_year + i) % 100 - 1:02d}-{(base_cap_year + i) % 100:02d}"
        for i in range(len(proj_df))
    ]

    base_value = contract["replacement_m"] + contract["dollars_per_spar_m"] * proj_df["Predicted_pSPAR"]
    proj_df["Market_Value_M"] = (base_value * (proj_df["Proj_Cap_M"] / BASE_CAP_M)).clip(lower=LEAGUE_MIN_SALARY)

    future = proj_df[proj_df["Season_Index"] != "Current"].reset_index(drop=True)
    contract_table = pd.DataFrame({
        "Term": range(1, 9),
        "Total_Value_M": future["Market_Value_M"].cumsum().round(2),
    })
    contract_table["AAV_M"] = (contract_table["Total_Value_M"] / contract_table["Term"]).round(3)

    # ── Top comps ──
    season_counts = dataset.groupby("Player")["Season"].nunique()
    experienced = season_counts[season_counts > 3].index
    comps_pool = cohort[cohort["Player"].isin(experienced)]
    top_comps = comps_pool.head(10)[
        ["Player", "Season", "Age", "pSPAR", "predict_all_toi", "base_weight"]
    ].copy()
    top_comps.columns = ["Player", "Season", "Age", "pSPAR", "TOI", "Similarity"]

    return proj_df, contract_table, top_comps

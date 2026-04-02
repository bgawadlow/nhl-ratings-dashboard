"""
NHL Ratings Model — Private module (not tracked in git).
Contains proprietary model constants, projection logic, and financial calculations.
"""

import numpy as np
import pandas as pd

# ── Model Constants ─────────────────────────────────────────────────────────
MARKET_COEFS = {
    "F": {"intercept": 0.0118168, "slope": 0.0073865},
    "D": {"intercept": 0.0118966, "slope": 0.0082860},
}

OVR_PARAMS = {"F": (77.5, 0.875), "D": (78.25, 1.075)}

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

# Known salary caps by season start year
KNOWN_CAPS = {2026: 95.5, 2027: 104.0, 2028: 113.5}


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


# ── Dataset Builder ─────────────────────────────────────────────────────────

def build_scaled_dataset(spar_df, draft_df):
    """Merge SPAR + draft info, compute deltas, per-60 rates & z-scores."""
    df = spar_df.copy()

    draft_map = draft_df.drop_duplicates(subset="Player_Key", keep="first")
    df = df.merge(
        draft_map[["Player_Key", "Draft Yr", "Draft Ov"]],
        left_on="Player", right_on="Player_Key", how="left",
    ).drop(columns=["Player_Key"])

    max_pick = df["Draft Yr"].map(DRAFT_MAX_LOOKUP)
    df["Draft_Ov_Clean"] = np.where(
        df["Draft Ov"].notna(),
        df["Draft Ov"],
        np.where(max_pick.notna(), max_pick + 1, 225),
    )
    df["Draft_Ov_Log"] = np.log(df["Draft_Ov_Clean"].astype(float))

    df["Season_Num"] = df["Season"].str[3:5].astype(int) + 2000

    for col in PERF_VARS + ["pSPAR"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

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

    df = df.sort_values(["Player", "Season_Num"])
    for var in PERF_VARS:
        if var in df.columns:
            df[f"d_{var}"] = df.groupby("Player")[var].diff().fillna(0)

    delta_vars = [f"d_{v}" for v in PERF_VARS if v in df.columns]
    all_vars = [v for v in PERF_VARS if v in df.columns] + delta_vars + ["Draft_Ov_Log"]

    def safe_scale(s):
        if s.std() == 0 or s.count() < 2:
            return pd.Series(0, index=s.index)
        return (s - s.mean()) / s.std()

    for var in all_vars:
        if var in df.columns:
            df[f"z_{var}"] = df.groupby("Season")[var].transform(safe_scale)

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
    coefs = MARKET_COEFS[t_pos]

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

    proj_df["Market_Share_Pct"] = coefs["intercept"] + coefs["slope"] * proj_df["Predicted_pSPAR"]
    proj_df["Market_Value_M"] = proj_df["Market_Share_Pct"] * proj_df["Proj_Cap_M"]

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
    coefs = MARKET_COEFS[t_pos]

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

    proj_df["Market_Share_Pct"] = coefs["intercept"] + coefs["slope"] * proj_df["Predicted_pSPAR"]
    proj_df["Market_Value_M"] = proj_df["Market_Share_Pct"] * proj_df["Proj_Cap_M"]

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

"""
NHL 26 Player Ratings Dashboard
================================
Streamlit app displaying player ratings, GAR, SPAR, per-60 projections,
and skating metrics across all seasons (2007-08 to 2025-26).
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHL 26 Ratings",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Mobile-friendly CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* Collapse sidebar on mobile by default */
@media (max-width: 768px) {
    [data-testid="stSidebar"] { min-width: 0 !important; width: 0 !important; }
    [data-testid="stSidebar"] > div { display: none; }
    .block-container { padding: 1rem 0.5rem !important; max-width: 100% !important; }
    h1 { font-size: 1.5rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
    /* Make tabs scrollable */
    [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; flex-wrap: nowrap; }
    [data-testid="stTabs"] [role="tab"] { white-space: nowrap; font-size: 0.85rem; }
    /* Make dataframes scroll horizontally */
    [data-testid="stDataFrame"] { overflow-x: auto !important; }
}
/* General table improvements */
[data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    gar = pd.read_csv(DATA_DIR / "gar.csv")
    spar = pd.read_csv(DATA_DIR / "spar.csv")
    skating = pd.read_csv(DATA_DIR / "skating.csv")
    draft_path = DATA_DIR / "draft_info.csv"
    if draft_path.exists():
        draft_info = pd.read_csv(draft_path)
        draft_info["Player_Key"] = (
            draft_info["Player"].str.upper().str.replace(" ", ".", regex=False)
        )
    else:
        draft_info = pd.DataFrame(columns=["Player", "Draft Yr", "Draft Ov", "Player_Key"])
    return ratings, gar, spar, skating, draft_info


ratings, gar, spar, skating, draft_info = load_data()

# ── Sidebar Filters ─────────────────────────────────────────────────────────
st.sidebar.title("Filters")

# Season filter — default to most recent
all_seasons = sorted(ratings["Season"].dropna().unique(), reverse=True)
current_season = all_seasons[0] if len(all_seasons) > 0 else None

selected_seasons = st.sidebar.multiselect(
    "Season",
    options=all_seasons,
    default=[current_season] if current_season else [],
)

# Position filter
selected_pos = st.sidebar.radio(
    "Position",
    options=["All", "F", "D"],
    horizontal=True,
)

# Team filter
all_teams = sorted(ratings["Team"].dropna().unique())
selected_teams = st.sidebar.multiselect(
    "Team",
    options=all_teams,
    default=[],
    placeholder="All Teams",
)

# Player search
search = st.sidebar.text_input("Search Player", placeholder="e.g. McDavid")

# Min TOI/GP
min_toi = st.sidebar.slider("Min TOI/GP", 0.0, 25.0, 0.0, 0.5)


# ── Apply Filters ────────────────────────────────────────────────────────────
def apply_filters(df):
    filtered = df.copy()
    if selected_seasons:
        filtered = filtered[filtered["Season"].isin(selected_seasons)]
    if selected_pos != "All":
        filtered = filtered[filtered["Position"] == selected_pos]
    if selected_teams:
        filtered = filtered[filtered["Team"].isin(selected_teams)]
    if search:
        filtered = filtered[
            filtered["Player"].str.contains(search, case=False, na=False)
        ]
    if "TOI/GP" in filtered.columns and min_toi > 0:
        filtered = filtered[filtered["TOI/GP"] >= min_toi]
    return filtered


filtered_ratings = apply_filters(ratings)

# ── Main Content ─────────────────────────────────────────────────────────────
st.title("NHL 26 Player Ratings")
st.caption(
    f"Showing {len(filtered_ratings):,} player-seasons "
    f"({len(filtered_ratings['Player'].unique()):,} unique players)"
)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ratings, tab_gar, tab_contract, tab_toi, tab_player = st.tabs(
    ["Ratings", "GAR / SPAR", "Contract Value", "TOI Adjustor", "Player Lookup"]
)

# ── Tab 1: Ratings Table ────────────────────────────────────────────────────
with tab_ratings:
    display_cols = [
        "Player", "Season", "Team", "Position", "OVR", "Off", "Def",
        "Draw", "Take", "pSPAR", "TOI/GP", "G", "PTS", "Hits", "Blocks",
        "TOI/PP", "EV G", "EV PTS", "FO",
    ]
    available = [c for c in display_cols if c in filtered_ratings.columns]
    df_show = filtered_ratings[available].sort_values("OVR", ascending=False)

    # Format numeric columns to 2 decimal places
    fmt = {
        col: "{:.2f}"
        for col in df_show.select_dtypes(include="number").columns
    }
    # OVR as whole number
    if "OVR" in fmt:
        fmt["OVR"] = "{:.0f}"
    # Keep integer-like columns as integers
    for col in ["G", "PTS", "Hits", "Blocks", "EV G", "EV PTS", "FO"]:
        if col in fmt:
            fmt[col] = "{:.0f}"

    st.dataframe(
        df_show.style.format(fmt, na_rep="—"),
        use_container_width=True,
        height=700,
        hide_index=True,
    )

# ── Tab 2: GAR / SPAR ──────────────────────────────────────────────────────
with tab_gar:
    metric_type = st.radio(
        "Metric", ["GAR", "SPAR"], horizontal=True, key="gar_spar_toggle"
    )

    if metric_type == "GAR":
        src = apply_filters(gar)
        val_cols = [
            "pEVO_GAR", "pEVD_GAR", "pPPO_GAR", "pSHD_GAR",
            "pTAKE_GAR", "pDRAW_GAR", "pGAR",
        ]
    else:
        src = apply_filters(spar)
        val_cols = [
            "pEVO_SPAR", "pEVD_SPAR", "pPPO_SPAR", "pSHD_SPAR",
            "pTAKE_SPAR", "pDRAW_SPAR", "pSPAR",
        ]

    gar_display = ["Player", "Season", "Team", "Position", "Age"] + val_cols
    available_gar = [c for c in gar_display if c in src.columns]
    total_col = val_cols[-1]  # pGAR or pSPAR
    df_gar = src[available_gar].sort_values(total_col, ascending=False)

    fmt_gar = {col: "{:.2f}" for col in val_cols if col in df_gar.columns}

    st.dataframe(
        df_gar.style.format(fmt_gar, na_rep="—"),
        use_container_width=True,
        height=700,
        hide_index=True,
    )

# ── Tab 3: Contract Value ──────────────────────────────────────────────────
with tab_contract:
    st.caption("Similarity-based aging curve projections & market value analysis")

    # ── Contract model constants ──
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

    @st.cache_data
    def build_scaled_dataset(_spar_df, _draft_df):
        """Merge SPAR + draft info, compute deltas, per-60 rates & z-scores."""
        df = _spar_df.copy()

        # Join draft info
        draft_map = _draft_df.drop_duplicates(subset="Player_Key", keep="first")
        df = df.merge(
            draft_map[["Player_Key", "Draft Yr", "Draft Ov"]],
            left_on="Player", right_on="Player_Key", how="left",
        ).drop(columns=["Player_Key"])

        # Clean draft logic
        max_pick = df["Draft Yr"].map(DRAFT_MAX_LOOKUP)
        df["Draft_Ov_Clean"] = np.where(
            df["Draft Ov"].notna(),
            df["Draft Ov"],
            np.where(max_pick.notna(), max_pick + 1, 225),
        )
        df["Draft_Ov_Log"] = np.log(df["Draft_Ov_Clean"].astype(float))

        # Parse season number for sorting
        df["Season_Num"] = df["Season"].str[3:5].astype(int) + 2000

        # Fill NAs in perf vars
        for col in PERF_VARS + ["pSPAR"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Compute EV TOI and per-60 rates (for v2)
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

        # Compute deltas (year-over-year change)
        df = df.sort_values(["Player", "Season_Num"])
        for var in PERF_VARS:
            if var in df.columns:
                df[f"d_{var}"] = df.groupby("Player")[var].diff().fillna(0)

        # Z-score within each season
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

    df_scaled = build_scaled_dataset(spar, draft_info)

    # ── Survival table for v2 ──
    @st.cache_data
    def build_survival_table(_df):
        """P(still playing next season | position, age, SPAR bucket)."""
        df = _df.copy()
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

        # Fallback: position-age average
        pos_age = (
            df.groupby(["Position", "Age"])["has_next"]
            .mean().reset_index(name="rate_avg")
        )
        surv = surv.merge(pos_age, on=["Position", "Age"], how="left")
        surv["rate"] = surv["rate"].fillna(surv["rate_avg"]).fillna(0.5)
        return surv

    survival_table = build_survival_table(df_scaled)

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

    # Salary cap lookup — known caps by season start year
    # Ratings labeled "25-26" are projections for 26-27, so "Current" = 26-27 cap
    KNOWN_CAPS = {2026: 95.5, 2027: 104.0, 2028: 113.5}

    def get_cap(season_start_year):
        """Return projected salary cap ($M) for a given season start year."""
        if season_start_year in KNOWN_CAPS:
            return KNOWN_CAPS[season_start_year]
        # 5% annual growth from 28-29 onward
        return round(113.5 * (1.05 ** (season_start_year - 2028)), 1)

    # ── v1 projection ──
    def get_projection(target_name, target_season, dataset):
        """Run similarity-based projection for a player."""
        target = dataset[(dataset["Player"] == target_name) & (dataset["Season"] == target_season)]
        if target.empty:
            return None, None, None
        target = target.iloc[0]
        t_age = target["Age"]
        t_pos = target["Position"]

        # Variable selection
        delta_vars = [f"d_{v}" for v in PERF_VARS if v in dataset.columns]
        active_vars = (
            [v for v in PERF_VARS if v in dataset.columns]
            + delta_vars
            + (["Draft_Ov_Log"] if t_age <= 26 else [])
        )
        z_vars = [f"z_{v}" for v in active_vars if f"z_{v}" in dataset.columns]

        target_stats = target[z_vars].values.astype(float)

        # Find cohort (same position & age, different player)
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

        # Build 8-year aging curve
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

        # Financial calculations
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

        # Salary cap projections — ratings are projections for next season
        base_cap_year = int(target["Season_Num"]) + 1  # "25-26" → projects 26-27
        proj_df["Proj_Cap_M"] = [get_cap(base_cap_year + i) for i in range(len(proj_df))]

        # Add actual season labels
        proj_df["Season_Label"] = [
            f"{(base_cap_year + i) % 100 - 1:02d}-{(base_cap_year + i) % 100:02d}"
            for i in range(len(proj_df))
        ]

        proj_df["Market_Share_Pct"] = coefs["intercept"] + coefs["slope"] * proj_df["Predicted_pSPAR"]
        proj_df["Market_Value_M"] = proj_df["Market_Share_Pct"] * proj_df["Proj_Cap_M"]

        # Contract term table
        future = proj_df[proj_df["Season_Index"] != "Current"].reset_index(drop=True)
        contract_table = pd.DataFrame({
            "Term": range(1, 9),
            "Total_Value_M": future["Market_Value_M"].cumsum().round(2),
        })
        contract_table["AAV_M"] = (contract_table["Total_Value_M"] / contract_table["Term"]).round(3)

        # Filter comps to players with >3 seasons of data
        season_counts = dataset.groupby("Player")["Season"].nunique()
        experienced = season_counts[season_counts > 3].index
        comps_pool = cohort[cohort["Player"].isin(experienced)]
        top_comps = comps_pool.head(5)[["Player", "Season", "Age", "pSPAR", "predict_all_toi", "weight"]].copy()
        top_comps.columns = ["Player", "Season", "Age", "pSPAR", "TOI", "Similarity"]

        return proj_df, contract_table, top_comps

    # ── v2 projection (component-level aging, decay, TOI-split, survival) ──
    def get_projection_v2(target_name, target_season, dataset, decay_rate=0.85,
                          era_decay=0.92):
        """v2: component-level aging, similarity decay, TOI-split, survival bias."""
        target = dataset[(dataset["Player"] == target_name) & (dataset["Season"] == target_season)]
        if target.empty:
            return None, None, None
        target = target.iloc[0]
        t_age = int(target["Age"])
        t_pos = target["Position"]

        # Similarity (same as v1)
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

        # Current per-60 rates and TOI
        rate_keys = ["rate_EVO", "rate_EVD", "rate_PPO", "rate_SHD", "rate_TAKE", "rate_DRAW"]
        curr_rates = {k: float(target[k]) for k in rate_keys}
        curr_toi = {
            "ev": float(target["ev_toi"]),
            "pp": float(target.get("predict_pp_toi", 0) or 0),
            "sh": float(target.get("predict_sh_toi", 0) or 0),
        }

        # 8-year component-level projection
        projections_list = []
        for i in range(1, 9):
            next_age = t_age + i

            # Find comp transitions
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
                # Era weighting: recent transitions count more
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

            # Update rates
            for rk in rate_keys:
                curr_rates[rk] += avg_d.get(f"d_{rk}", 0)

            # Update TOI (floor at 0)
            curr_toi["ev"] = max(curr_toi["ev"] + avg_d.get("d_ev_toi", 0), 0)
            curr_toi["pp"] = max(curr_toi["pp"] + avg_d.get("d_pp_toi", 0), 0)
            curr_toi["sh"] = max(curr_toi["sh"] + avg_d.get("d_sh_toi", 0), 0)
            all_toi = curr_toi["ev"] + curr_toi["pp"] + curr_toi["sh"]

            # Recombine: component = (rate/60) * toi * 82
            comp_vals = {
                "pEVO": (curr_rates["rate_EVO"] / 60) * curr_toi["ev"] * 82,
                "pEVD": (curr_rates["rate_EVD"] / 60) * curr_toi["ev"] * 82,
                "pPPO": (curr_rates["rate_PPO"] / 60) * curr_toi["pp"] * 82,
                "pSHD": (curr_rates["rate_SHD"] / 60) * curr_toi["sh"] * 82,
                "pTAKE": (curr_rates["rate_TAKE"] / 60) * all_toi * 82,
                "pDRAW": (curr_rates["rate_DRAW"] / 60) * all_toi * 82,
            }
            raw_spar = sum(comp_vals.values())

            # Survival probability (informational — not applied to projections)
            surv_prob = get_survival_prob(t_pos, next_age - 1, raw_spar, survival_table)

            projections_list.append({
                "Age": next_age,
                "Predicted_pSPAR": raw_spar,
                "Survival_Prob": surv_prob,
                "TOI": round(all_toi, 1),
                "Season_Index": f"+{i}",
            })

        # Financials
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

        # Salary cap projections — ratings are projections for next season
        base_cap_year = int(target["Season_Num"]) + 1
        proj_df["Proj_Cap_M"] = [get_cap(base_cap_year + i) for i in range(len(proj_df))]

        # Add actual season labels
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
        top_comps = comps_pool.head(5)[["Player", "Season", "Age", "pSPAR", "predict_all_toi", "base_weight"]].copy()
        top_comps.columns = ["Player", "Season", "Age", "pSPAR", "TOI", "Similarity"]

        return proj_df, contract_table, top_comps

    # ── UI ──
    cv_col1, cv_col2 = st.columns([2, 1])
    with cv_col1:
        cv_players = sorted(spar[spar["Season"] == (all_seasons[0] if all_seasons else "25-26")]["Player"].unique())
        cv_player = st.selectbox("Select Player", cv_players, index=None, placeholder="Type to search...", key="cv_player")
    with cv_col2:
        cv_season = st.selectbox("Season", all_seasons, index=0, key="cv_season")

    if cv_player:
        proj_df, contract_table, top_comps = get_projection(cv_player, cv_season, df_scaled)

        if proj_df is not None:
            # Headline
            current = proj_df[proj_df["Season_Index"] == "Current"].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Current pSPAR", f"{current['Predicted_pSPAR']:.2f}")
            c2.metric("Current OVR", f"{current['Predicted_OVR']}")
            pos = spar[(spar["Player"] == cv_player) & (spar["Season"] == cv_season)]["Position"].iloc[0]
            c3.metric("Position", pos)

            # Projection table
            st.markdown("**8-Year Aging Curve Projection**")
            proj_display = proj_df[["Season_Label", "Age", "Predicted_pSPAR", "Predicted_OVR", "Proj_Cap_M", "Market_Value_M"]].copy()
            proj_display.columns = ["Season", "Age", "pSPAR", "OVR", "Proj Cap ($M)", "Market Value ($M)"]
            fmt_proj_cv = {"pSPAR": "{:.2f}", "Proj Cap ($M)": "${:.1f}M", "Market Value ($M)": "${:.2f}M"}
            st.dataframe(proj_display.style.format(fmt_proj_cv, na_rep="—"), use_container_width=True, hide_index=True)

            # Contract term AAV table
            st.markdown("**Fair Market Value by Term Length**")
            ct_display = contract_table.copy()
            ct_display.columns = ["Term (Years)", "Total Value ($M)", "AAV ($M)"]
            fmt_ct = {"Total Value ($M)": "${:.2f}M", "AAV ($M)": "${:.3f}M"}
            st.dataframe(ct_display.style.format(fmt_ct), use_container_width=True, hide_index=True)

            # Contract proposal analyzer
            st.markdown("---")
            st.markdown("**Evaluate a Contract Proposal**")
            prop_col1, prop_col2 = st.columns(2)
            with prop_col1:
                user_aav = st.number_input("Proposed AAV ($M)", min_value=0.0, max_value=20.0, value=5.0, step=0.25, key="cv_aav")
            with prop_col2:
                user_term = st.slider("Term (Years)", 1, 8, 4, key="cv_term")

            future = proj_df[proj_df["Season_Index"] != "Current"].head(user_term).copy()
            future["Surplus"] = future["Market_Value_M"] - user_aav
            total_surplus = future["Surplus"].sum()

            if total_surplus > 0:
                st.success(f"BARGAIN — Total surplus: **${total_surplus:.2f}M** over {user_term} years")
            else:
                st.error(f"OVERPAY — Total deficit: **${total_surplus:.2f}M** over {user_term} years")

            # Surplus chart
            chart_df = future[["Age", "Market_Value_M"]].set_index("Age")
            chart_df["Proposed AAV"] = user_aav
            chart_df.columns = ["Market Value", "Proposed AAV"]
            st.line_chart(chart_df)

            # Year-by-year breakdown
            breakdown = future[["Season_Label", "Age", "Predicted_OVR", "Market_Value_M", "Surplus"]].copy()
            breakdown.columns = ["Year", "Age", "Proj OVR", "Market Value ($M)", "Surplus ($M)"]
            fmt_bd = {"Market Value ($M)": "${:.2f}M", "Surplus ($M)": "${:.2f}M"}
            st.dataframe(breakdown.style.format(fmt_bd), use_container_width=True, hide_index=True)

            # Top comps
            if top_comps is not None and not top_comps.empty:
                st.markdown("**Closest Comparables**")
                comp_fmt = {"pSPAR": "{:.1f}", "TOI": "{:.1f}", "Similarity": "{:.3f}"}
                st.dataframe(top_comps.style.format(comp_fmt, na_rep="—"), use_container_width=True, hide_index=True)
        else:
            st.warning(f"No data found for {cv_player} in {cv_season}")

    # ── Aging Curve by Era (diagnostic) ──
    st.markdown("---")
    with st.expander("Aging Curves by Era (diagnostic)", expanded=False):
        st.caption(
            "Year-over-year pSPAR delta by age, split by era. Uses each player as "
            "their own control to isolate the aging effect (avoids survivorship bias). "
            "Peak age = where cumulative curve is maximized."
        )
        era_pos = st.radio("Position", ["F", "D"], horizontal=True, key="era_pos")
        era_min_toi = st.slider("Min TOI/GP", 0.0, 15.0, 8.0, 0.5, key="era_min_toi",
                                help="Filter to regulars only")

        era_df = df_scaled[
            (df_scaled["Position"] == era_pos)
            & (df_scaled["predict_all_toi"] >= era_min_toi)
            & (df_scaled["Age"] >= 19)
            & (df_scaled["Age"] <= 40)
        ].copy()

        # Compute same-player year-over-year delta
        era_df = era_df.sort_values(["Player", "Season_Num"])
        era_df["prev_pSPAR"] = era_df.groupby("Player")["pSPAR"].shift(1)
        era_df["prev_Age"] = era_df.groupby("Player")["Age"].shift(1)
        era_df["prev_Season"] = era_df.groupby("Player")["Season_Num"].shift(1)

        # Only keep consecutive-season transitions (age +1, season +1)
        delta_df = era_df[
            (era_df["Age"] == era_df["prev_Age"] + 1)
            & (era_df["Season_Num"] == era_df["prev_Season"] + 1)
        ].copy()
        delta_df["delta_pSPAR"] = delta_df["pSPAR"] - delta_df["prev_pSPAR"]

        # Era is assigned by the destination season
        delta_df["Era"] = pd.cut(
            delta_df["Season_Num"],
            bins=[2006, 2013, 2019, 2030],
            labels=["07-13", "14-19", "20-26"],
        )

        # Average delta by age transition and era (min 15 observations)
        delta_agg = (
            delta_df.groupby(["Era", "Age"])["delta_pSPAR"]
            .agg(["mean", "count"])
            .reset_index()
        )
        delta_curve = delta_agg[delta_agg["count"] >= 15].copy()
        delta_curve.rename(columns={"mean": "Avg Delta"}, inplace=True)

        if not delta_curve.empty:
            # Plot average delta by age
            st.markdown("**Average Year-over-Year pSPAR Change by Age**")
            delta_chart = delta_curve.pivot(index="Age", columns="Era", values="Avg Delta")
            st.line_chart(delta_chart)

            # Build cumulative curve to find peak age
            st.markdown("**Cumulative Aging Curve (indexed to age 21)**")
            cumul_rows = []
            for era in delta_curve["Era"].unique():
                era_data = delta_curve[delta_curve["Era"] == era].sort_values("Age")
                cumul = 0.0
                for _, row in era_data.iterrows():
                    cumul += row["Avg Delta"]
                    cumul_rows.append({"Era": era, "Age": int(row["Age"]), "Cumulative": round(cumul, 3)})
            cumul_df = pd.DataFrame(cumul_rows)
            cumul_chart = cumul_df.pivot(index="Age", columns="Era", values="Cumulative")
            st.line_chart(cumul_chart)

            # Peak age = where cumulative curve is maximized
            peaks = cumul_df.loc[cumul_df.groupby("Era")["Cumulative"].idxmax()][["Era", "Age", "Cumulative"]]
            peaks.columns = ["Era", "Peak Age", "Cumul. Delta at Peak"]
            peaks["Cumul. Delta at Peak"] = peaks["Cumul. Delta at Peak"].round(2)
            st.dataframe(peaks, use_container_width=True, hide_index=True)

            # Show sample sizes
            with st.expander("Sample sizes by age/era"):
                size_tbl = delta_agg.pivot(index="Age", columns="Era", values="count").fillna(0).astype(int)
                st.dataframe(size_tbl, use_container_width=True)
        else:
            st.info("Not enough data for the selected filters.")

# ── Tab 5: TOI Adjustor ────────────────────────────────────────────────────
with tab_toi:
    st.caption(
        "Adjust a player's EV / PP / SH ice time and see how their SPAR & OVR change. "
        "Per-60 rates stay fixed — only usage changes."
    )

    # OVR formula
    OVR_F = lambda spar_val: 77.5 + 0.875 * spar_val
    OVR_D = lambda spar_val: 78.25 + 1.075 * spar_val

    # Player selection (current season only from SPAR)
    curr = all_seasons[0] if all_seasons else "25-26"
    spar_curr = spar[spar["Season"] == curr].copy()

    # Compute EV TOI
    spar_curr["EV_TOI"] = (
        spar_curr["predict_all_toi"]
        - spar_curr["predict_pp_toi"].fillna(0)
        - spar_curr["predict_sh_toi"].fillna(0)
    )

    toi_players = sorted(spar_curr["Player"].unique())
    toi_player = st.selectbox(
        "Select Player", toi_players, index=None,
        placeholder="Type to search...", key="toi_player",
    )

    if toi_player:
        row = spar_curr[spar_curr["Player"] == toi_player].iloc[0]
        pos = row["Position"]

        old_ev = float(row["EV_TOI"])
        old_pp = float(row.get("predict_pp_toi", 0) or 0)
        old_sh = float(row.get("predict_sh_toi", 0) or 0)
        old_all = float(row["predict_all_toi"])

        # Current per-60 above-replacement rates (back-calculated)
        # component = (rate/60) * toi * 82  →  rate/60 = component * 60 / (toi * 82)
        def back_calc_rate(component, toi):
            if toi == 0 or pd.isna(toi) or pd.isna(component):
                return 0.0
            return float(component) * 60 / (toi * 82)

        rates = {
            "pEVO_SPAR": back_calc_rate(row["pEVO_SPAR"], old_ev),
            "pEVD_SPAR": back_calc_rate(row["pEVD_SPAR"], old_ev),
            "pPPO_SPAR": back_calc_rate(row["pPPO_SPAR"], old_pp),
            "pSHD_SPAR": back_calc_rate(row["pSHD_SPAR"], old_sh),
            "pTAKE_SPAR": back_calc_rate(row["pTAKE_SPAR"], old_all),
            "pDRAW_SPAR": back_calc_rate(row["pDRAW_SPAR"], old_all),
        }

        # Show current state
        st.markdown(f"**{toi_player}** — {pos} — Current TOI/GP: {old_all:.1f} min")

        st.markdown("---")
        st.markdown("**Adjust Ice Time (minutes per game)**")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            new_ev = st.slider(
                "EV TOI/GP", 0.0, 25.0, round(old_ev, 1), 0.5, key="toi_ev",
            )
        with tc2:
            new_pp = st.slider(
                "PP TOI/GP", 0.0, 8.0, round(old_pp, 1), 0.25, key="toi_pp",
            )
        with tc3:
            new_sh = st.slider(
                "SH TOI/GP", 0.0, 6.0, round(old_sh, 1), 0.25, key="toi_sh",
            )
        new_all = new_ev + new_pp + new_sh

        # Recalculate SPAR components with new TOI
        new_components = {
            "EV Off (pEVO)": rates["pEVO_SPAR"] / 60 * new_ev * 82,
            "EV Def (pEVD)": rates["pEVD_SPAR"] / 60 * new_ev * 82,
            "PP Off (pPPO)": rates["pPPO_SPAR"] / 60 * new_pp * 82,
            "SH Def (pSHD)": rates["pSHD_SPAR"] / 60 * new_sh * 82,
            "Takeaways (pTAKE)": rates["pTAKE_SPAR"] / 60 * new_all * 82,
            "Drawing (pDRAW)": rates["pDRAW_SPAR"] / 60 * new_all * 82,
        }
        new_spar = sum(new_components.values())
        old_spar = float(row["pSPAR"])

        ovr_fn = OVR_F if pos == "F" else OVR_D
        new_ovr = round(ovr_fn(new_spar))
        old_ovr = round(ovr_fn(old_spar))

        # Display results
        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("New TOI/GP", f"{new_all:.1f}", f"{new_all - old_all:+.1f}")
        r2.metric("New pSPAR", f"{new_spar:.2f}", f"{new_spar - old_spar:+.2f}")
        r3.metric("New OVR", f"{new_ovr}", f"{new_ovr - old_ovr:+d}")
        r4.metric("Position", pos)

        # Component breakdown table
        st.markdown("**SPAR Component Breakdown**")
        old_components = {
            "EV Off (pEVO)": float(row["pEVO_SPAR"]),
            "EV Def (pEVD)": float(row["pEVD_SPAR"]),
            "PP Off (pPPO)": float(row["pPPO_SPAR"]),
            "SH Def (pSHD)": float(row["pSHD_SPAR"]),
            "Takeaways (pTAKE)": float(row["pTAKE_SPAR"]),
            "Drawing (pDRAW)": float(row["pDRAW_SPAR"]),
        }
        comp_df = pd.DataFrame({
            "Component": list(old_components.keys()),
            "Original": list(old_components.values()),
            "Adjusted": list(new_components.values()),
            "Change": [new_components[k] - old_components[k] for k in old_components],
        })
        comp_df["TOI Used"] = ["EV", "EV", "PP", "SH", "Total", "Total"]
        fmt_comp = {"Original": "{:.2f}", "Adjusted": "{:.2f}", "Change": "{:+.2f}"}
        st.dataframe(
            comp_df.style.format(fmt_comp, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        # Per-60 rates reference
        st.markdown("**Per-60 Above-Average Rates (Fixed)**")
        rate_labels = {
            "pEVO_SPAR": "EV Off/60", "pEVD_SPAR": "EV Def/60",
            "pPPO_SPAR": "PP Off/60", "pSHD_SPAR": "SH Def/60",
            "pTAKE_SPAR": "Take/60", "pDRAW_SPAR": "Draw/60",
        }
        rate_df = pd.DataFrame({
            "Metric": list(rate_labels.values()),
            "Per-60 Rate": [rates[k] for k in rate_labels],
        })
        st.dataframe(
            rate_df.style.format({"Per-60 Rate": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )

# ── Tab 6: Player Lookup ────────────────────────────────────────────────────
with tab_player:
    all_players = sorted(ratings["Player"].dropna().unique())
    selected_player = st.selectbox(
        "Select a player",
        options=all_players,
        index=None,
        placeholder="Type to search...",
    )

    if selected_player:
        st.subheader(selected_player)

        # ── Ratings history ──
        player_ratings = (
            ratings[ratings["Player"] == selected_player]
            .sort_values("Season", ascending=False)
        )
        if not player_ratings.empty:
            latest = player_ratings.iloc[0]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("OVR", f"{latest['OVR']:.0f}")
            col2.metric("Off", f"{latest['Off']:.2f}")
            col3.metric("Def", f"{latest['Def']:.2f}")
            col4.metric("pSPAR", f"{latest['pSPAR']:.2f}")
            col5.metric(
                "TOI/GP",
                f"{latest['TOI/GP']:.1f}" if pd.notna(latest["TOI/GP"]) else "—",
            )

            st.markdown("**Season-by-Season Ratings**")
            hist_cols = [
                "Season", "Team", "Position", "OVR", "Off", "Def",
                "Draw", "Take", "pSPAR", "TOI/GP", "G", "PTS",
            ]
            available_hist = [c for c in hist_cols if c in player_ratings.columns]
            fmt_hist = {
                col: "{:.2f}"
                for col in player_ratings[available_hist]
                .select_dtypes(include="number")
                .columns
            }
            for col in ["OVR", "G", "PTS"]:
                if col in fmt_hist:
                    fmt_hist[col] = "{:.0f}"
            st.dataframe(
                player_ratings[available_hist].style.format(fmt_hist, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )

        # ── OVR trend chart ──
        if len(player_ratings) > 1:
            import altair as alt
            st.markdown("**OVR Trend**")
            chart_data = player_ratings[["Season", "OVR"]].copy()
            ovr_chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Season:N", sort=None),
                    y=alt.Y("OVR:Q", scale=alt.Scale(domain=[65, 100])),
                    tooltip=["Season", "OVR"],
                )
                .properties(height=300)
            )
            st.altair_chart(ovr_chart, use_container_width=True)

        # ── GAR breakdown (latest season) ──
        player_gar = gar[gar["Player"] == selected_player].sort_values(
            "Season", ascending=False
        )
        if not player_gar.empty:
            latest_gar = player_gar.iloc[0]
            st.markdown(f"**GAR Breakdown** ({latest_gar['Season']})")
            gar_components = {
                "EV Offense": latest_gar.get("pEVO_GAR", 0),
                "EV Defense": latest_gar.get("pEVD_GAR", 0),
                "PP Offense": latest_gar.get("pPPO_GAR", 0),
                "SH Defense": latest_gar.get("pSHD_GAR", 0),
                "Takeaways": latest_gar.get("pTAKE_GAR", 0),
                "Drawing": latest_gar.get("pDRAW_GAR", 0),
            }
            st.bar_chart(pd.Series(gar_components))
            st.metric("Total pGAR", f"{latest_gar.get('pGAR', 0):.2f}")

        # ── Skating ──
        player_skating = skating[skating["Player"] == selected_player]
        if not player_skating.empty:
            latest_sk = player_skating.sort_values(
                "Season", ascending=False
            ).iloc[0]
            st.markdown(f"**Skating Metrics** ({latest_sk['Season']})")
            sk1, sk2 = st.columns(2)
            blend_18 = latest_sk.get("18-20 Blend")
            blend_20 = latest_sk.get("20-22+ Blend")
            if pd.notna(blend_18):
                sk1.metric("18-20 Game Blend", f"{blend_18:.0f}")
            if pd.notna(blend_20):
                sk2.metric("20-22+ Game Blend", f"{blend_20:.0f}")
            st.caption("100 = league average")

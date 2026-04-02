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
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    gar = pd.read_csv(DATA_DIR / "gar.csv")
    spar = pd.read_csv(DATA_DIR / "spar.csv")
    projections = pd.read_csv(DATA_DIR / "projections.csv")
    skating = pd.read_csv(DATA_DIR / "skating.csv")
    draft_path = DATA_DIR / "draft_info.csv"
    if draft_path.exists():
        draft_info = pd.read_csv(draft_path)
        draft_info["Player_Key"] = (
            draft_info["Player"].str.upper().str.replace(" ", ".", regex=False)
        )
    else:
        draft_info = pd.DataFrame(columns=["Player", "Draft Yr", "Draft Ov", "Player_Key"])
    return ratings, gar, spar, projections, skating, draft_info


ratings, gar, spar, projections, skating, draft_info = load_data()

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
tab_ratings, tab_gar, tab_projections, tab_contract, tab_player = st.tabs(
    ["Ratings", "GAR / SPAR", "Per-60 Projections", "Contract Value", "Player Lookup"]
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

# ── Tab 3: Per-60 Projections ───────────────────────────────────────────────
with tab_projections:
    st.caption("Current season (25-26) per-60-minute rate projections")

    proj_filtered = projections.copy()
    if selected_pos != "All":
        proj_filtered = proj_filtered[proj_filtered["Position"] == selected_pos]
    if selected_teams:
        proj_filtered = proj_filtered[proj_filtered["Team"].isin(selected_teams)]
    if search:
        proj_filtered = proj_filtered[
            proj_filtered["Player"].str.contains(search, case=False, na=False)
        ]

    proj_cols = [
        "Player", "Position", "Team", "TOI/GP", "EV TOI/GP",
        "PP TOI/GP", "SH TOI/GP", "EVO/60", "EVD/60", "PPO/60",
        "SHD/60", "TAKE/60", "DRAW/60",
    ]
    available_proj = [c for c in proj_cols if c in proj_filtered.columns]
    df_proj = proj_filtered[available_proj].sort_values(
        "TOI/GP", ascending=False
    )

    fmt_proj = {
        col: "{:.2f}"
        for col in df_proj.select_dtypes(include="number").columns
    }

    st.dataframe(
        df_proj.style.format(fmt_proj, na_rep="—"),
        use_container_width=True,
        height=700,
        hide_index=True,
    )

# ── Tab 4: Contract Value ──────────────────────────────────────────────────
with tab_contract:
    st.caption("Similarity-based aging curve projections & market value analysis")

    # ── Contract model constants ──
    MARKET_COEFS = {
        "F": {"intercept": 0.0118168, "slope": 0.0073865},
        "D": {"intercept": 0.0118966, "slope": 0.0082860},
    }
    OVR_PARAMS = {"F": (77.5, 0.9), "D": (78.25, 1.1)}

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
        """Merge SPAR + draft info, compute deltas & z-scores."""
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

        # Salary cap projections
        caps = [95.5, 104.0, 113.5]  # Current, +1, +2
        for i in range(3, 9):
            caps.append(round(113.5 * (1.05 ** (i - 2)), 1))
        proj_df["Proj_Cap_M"] = caps[: len(proj_df)]

        proj_df["Market_Share_Pct"] = coefs["intercept"] + coefs["slope"] * proj_df["Predicted_pSPAR"]
        proj_df["Market_Value_M"] = proj_df["Market_Share_Pct"] * proj_df["Proj_Cap_M"]

        # Contract term table
        future = proj_df[proj_df["Season_Index"] != "Current"].reset_index(drop=True)
        contract_table = pd.DataFrame({
            "Term": range(1, 9),
            "Total_Value_M": future["Market_Value_M"].cumsum().round(2),
        })
        contract_table["AAV_M"] = (contract_table["Total_Value_M"] / contract_table["Term"]).round(3)

        top_comps = cohort.head(5)[["Player", "Season", "Age", "pSPAR", "predict_all_toi", "weight"]].copy()
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
            proj_display = proj_df[["Season_Index", "Age", "Predicted_pSPAR", "Predicted_OVR", "Proj_Cap_M", "Market_Value_M"]].copy()
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
            breakdown = future[["Season_Index", "Age", "Predicted_OVR", "Market_Value_M", "Surplus"]].copy()
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

# ── Tab 5: Player Lookup ────────────────────────────────────────────────────
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
            for col in ["G", "PTS"]:
                if col in fmt_hist:
                    fmt_hist[col] = "{:.0f}"
            st.dataframe(
                player_ratings[available_hist].style.format(fmt_hist, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )

        # ── OVR trend chart ──
        if len(player_ratings) > 1:
            st.markdown("**OVR Trend**")
            chart_data = player_ratings[["Season", "OVR"]].set_index("Season")
            st.line_chart(chart_data)

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

        # ── Per-60 rates ──
        player_proj = projections[projections["Player"] == selected_player]
        if not player_proj.empty:
            p60 = player_proj.iloc[0]
            st.markdown("**Per-60 Rates (25-26 Projection)**")
            rate_cols = ["EVO/60", "EVD/60", "PPO/60", "SHD/60", "TAKE/60", "DRAW/60"]
            available_rates = [c for c in rate_cols if c in player_proj.columns]
            rates = {col: p60[col] for col in available_rates if pd.notna(p60[col])}
            if rates:
                st.bar_chart(pd.Series(rates))

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

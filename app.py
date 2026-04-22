"""
NHL 26 Player Ratings Dashboard
================================
Streamlit app displaying player ratings, GAR, SPAR, per-60 projections,
and skating metrics across all seasons (2007-08 to 2025-26).
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

# ── Model import (private, gitignored) ────────────────────────────────────
try:
    from model import (
        CONTRACT_MODEL, BASE_CAP_M, OVR_PARAMS, PERF_VARS, DRAFT_MAX_LOOKUP, KNOWN_CAPS,
        build_scaled_dataset, build_survival_table, build_survival_model,
        get_survival_prob, get_survival_prob_v3,
        get_projection, get_projection_v3,
        get_cap, ovr_f, ovr_d, compute_ovr,
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Build tag: pSPAR bounds v2 (2026-04-20)
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

# Clear all cached data on every fresh deploy so new model logic takes effect
# immediately without waiting for TTL expiry. No-op after the first run.
if "cache_cleared_v2" not in st.session_state:
    st.cache_data.clear()
    st.session_state["cache_cleared_v2"] = True


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=300)  # TODO: bump back to 3600 after confirming refresh
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
tab_ratings, tab_gar, tab_contract, tab_toi, tab_player, tab_compare = st.tabs(
    ["Ratings", "GAR / SPAR", "Contract Value", "TOI Adjustor", "Player Lookup", "Compare Players"]
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

    if not MODEL_AVAILABLE:
        st.warning("Contract projection model is not available in this deployment.")
    else:
        # ── Cached wrappers around model functions ──
        @st.cache_data(ttl=300)  # TODO: bump back to 3600 after confirming refresh
        def _build_scaled_dataset(_spar_df, _draft_df):
            return build_scaled_dataset(_spar_df, _draft_df)

        @st.cache_data(ttl=300)  # TODO: bump back to 3600 after confirming refresh
        def _build_survival_model(_df):
            return build_survival_model(_df)

        df_scaled = _build_scaled_dataset(spar, draft_info)
        survival_model = _build_survival_model(df_scaled)

        # ── UI ──
        cv_col1, cv_col2 = st.columns([2, 1])
        with cv_col1:
            cv_players = sorted(spar[spar["Season"] == (all_seasons[0] if all_seasons else "25-26")]["Player"].unique())
            cv_player = st.selectbox("Select Player", cv_players, index=None, placeholder="Type to search...", key="cv_player")
        with cv_col2:
            cv_season = st.selectbox("Season", all_seasons, index=0, key="cv_season")

        if cv_player:
            proj_df, contract_table, top_comps = get_projection_v3(cv_player, cv_season, df_scaled, survival_model)

            if proj_df is not None:
                # Headline
                current = proj_df[proj_df["Season_Index"] == "Current"].iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Current pSPAR", f"{current['Predicted_pSPAR']:.2f}")
                c2.metric("Current OVR", f"{current['Predicted_OVR']}")
                pos = spar[(spar["Player"] == cv_player) & (spar["Season"] == cv_season)]["Position"].iloc[0]
                c3.metric("Position", pos)

                # Projection table (with P10/P90 range from weighted comp distribution)
                st.markdown("**8-Year Aging Curve Projection**")
                base_cols = ["Season_Label", "Age", "Predicted_pSPAR", "Predicted_OVR", "TOI", "Survival_Prob", "Proj_Cap_M", "Market_Value_M"]
                has_bounds = "pSPAR_Low" in proj_df.columns and "pSPAR_High" in proj_df.columns
                if has_bounds:
                    st.caption("pSPAR Range shows ±1 SD of comparable players' trajectories (~68% of outcomes) — wider = more uncertainty. OVR Range converts those bounds via the position's OVR formula.")
                    proj_display = proj_df[base_cols + ["pSPAR_Low", "pSPAR_High"]].copy()
                    # Convert pSPAR bounds to OVR bounds using the same formula as Predicted_OVR
                    o_base, o_mult = OVR_PARAMS[pos]
                    proj_display["OVR_Low"] = (o_base + o_mult * proj_display["pSPAR_Low"]).round(0)
                    proj_display["OVR_High"] = (o_base + o_mult * proj_display["pSPAR_High"]).round(0)

                    def _range_str(low, high):
                        if pd.isna(low) or pd.isna(high) or low == high:
                            return "—"
                        return f"{low:.1f} – {high:.1f}"
                    def _ovr_range_str(low, high):
                        if pd.isna(low) or pd.isna(high) or low == high:
                            return "—"
                        return f"{int(low)} – {int(high)}"
                    proj_display["pSPAR Range"] = proj_display.apply(
                        lambda r: _range_str(r["pSPAR_Low"], r["pSPAR_High"]), axis=1)
                    proj_display["OVR Range"] = proj_display.apply(
                        lambda r: _ovr_range_str(r["OVR_Low"], r["OVR_High"]), axis=1)
                    proj_display = proj_display[["Season_Label", "Age", "Predicted_pSPAR", "pSPAR Range",
                                                 "Predicted_OVR", "OVR Range",
                                                 "TOI", "Survival_Prob", "Proj_Cap_M", "Market_Value_M"]]
                    proj_display.columns = ["Season", "Age", "pSPAR", "pSPAR Range (±1 SD)",
                                            "OVR", "OVR Range (±1 SD)",
                                            "TOI", "Survival %", "Proj Cap ($M)", "Market Value ($M)"]
                else:
                    proj_display = proj_df[base_cols].copy()
                    proj_display.columns = ["Season", "Age", "pSPAR", "OVR", "TOI", "Survival %", "Proj Cap ($M)", "Market Value ($M)"]
                fmt_proj_cv = {"pSPAR": "{:.2f}", "TOI": "{:.1f}", "Survival %": "{:.0%}", "Proj Cap ($M)": "${:.1f}M", "Market Value ($M)": "${:.2f}M"}
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
                breakdown = future[["Season_Label", "Age", "Predicted_OVR", "Survival_Prob", "Proj_Cap_M", "Market_Value_M", "Surplus"]].copy()
                breakdown.columns = ["Year", "Age", "Proj OVR", "Survival %", "Proj Cap ($M)", "Market Value ($M)", "Surplus ($M)"]
                fmt_bd = {"Survival %": "{:.0%}", "Proj Cap ($M)": "${:.1f}M", "Market Value ($M)": "${:.2f}M", "Surplus ($M)": "${:.2f}M"}
                st.dataframe(breakdown.style.format(fmt_bd), use_container_width=True, hide_index=True)

                # Top comps
                if top_comps is not None and not top_comps.empty:
                    st.markdown("**Closest Comparables**")
                    comp_fmt = {"pSPAR": "{:.1f}", "TOI": "{:.1f}", "Similarity": "{:.3f}"}
                    st.dataframe(top_comps.style.format(comp_fmt, na_rep="—"), use_container_width=True, hide_index=True)
            else:
                st.warning(f"No data found for {cv_player} in {cv_season}")


# ── Tab 5: TOI Adjustor ────────────────────────────────────────────────────
with tab_toi:
    if not MODEL_AVAILABLE:
        st.warning("TOI Adjustor requires the model module, which is not available in this deployment.")
    else:
        st.caption(
            "Adjust a player's EV / PP / SH ice time and see how their SPAR & OVR change. "
            "Per-60 rates stay fixed — only usage changes."
        )

        # Player selection (current season only from SPAR)
        curr = all_seasons[0] if all_seasons else "25-26"
        spar_curr = spar[spar["Season"] == curr].copy()

        # Compute EV TOI
        spar_curr["EV_TOI"] = (
            spar_curr["predict_all_toi"]
            - spar_curr["predict_pp_toi"].fillna(0)
            - spar_curr["predict_sh_toi"].fillna(0)
        )

        # ── Compute role-average TOI benchmarks by position ──
        @st.cache_data
        def _compute_role_benchmarks(_df):
            """Rank players within each team by EV/PP/SH TOI and compute role averages."""
            benchmarks = {}
            for pos in ["F", "D"]:
                pos_df = _df[_df["Position"] == pos].copy()
                if pos == "F":
                    ev_sizes = [3, 3, 3, 3]
                    ev_labels = ["1st Line", "2nd Line", "3rd Line", "4th Line"]
                    st_sizes = [3, 3]
                    st_labels_pp = ["PP1", "PP2", "Other"]
                    st_labels_sh = ["SH1", "SH2", "Other"]
                else:
                    ev_sizes = [2, 2, 2]
                    ev_labels = ["1st Pair", "2nd Pair", "3rd Pair"]
                    st_sizes = [2, 2]
                    st_labels_pp = ["PP1", "PP2", "Other"]
                    st_labels_sh = ["SH1", "SH2", "Other"]

                # EV roles
                pos_df["ev_rank"] = pos_df.groupby("Team")["EV_TOI"].rank(ascending=False, method="first")
                ev_avg = {}
                cutoff = 0
                for i, (size, label) in enumerate(zip(ev_sizes, ev_labels)):
                    low, high = cutoff + 1, cutoff + size
                    role_data = pos_df[(pos_df["ev_rank"] >= low) & (pos_df["ev_rank"] <= high)]["EV_TOI"]
                    ev_avg[label] = round(role_data.mean(), 1) if len(role_data) > 0 else 0.0
                    cutoff = high

                # PP roles
                pp_players = pos_df[pos_df["predict_pp_toi"].fillna(0) > 0.5].copy()
                pp_players["pp_rank"] = pp_players.groupby("Team")["predict_pp_toi"].rank(ascending=False, method="first")
                pp_avg = {}
                cutoff = 0
                for size, label in zip(st_sizes, st_labels_pp[:-1]):
                    low, high = cutoff + 1, cutoff + size
                    role_data = pp_players[(pp_players["pp_rank"] >= low) & (pp_players["pp_rank"] <= high)]["predict_pp_toi"]
                    pp_avg[label] = round(role_data.mean(), 1) if len(role_data) > 0 else 0.0
                    cutoff = high
                other_pp = pp_players[pp_players["pp_rank"] > cutoff]["predict_pp_toi"]
                pp_avg["Other"] = round(other_pp.mean(), 1) if len(other_pp) > 0 else 0.0

                # SH roles
                sh_players = pos_df[pos_df["predict_sh_toi"].fillna(0) > 0.3].copy()
                sh_players["sh_rank"] = sh_players.groupby("Team")["predict_sh_toi"].rank(ascending=False, method="first")
                sh_avg = {}
                cutoff = 0
                for size, label in zip(st_sizes, st_labels_sh[:-1]):
                    low, high = cutoff + 1, cutoff + size
                    role_data = sh_players[(sh_players["sh_rank"] >= low) & (sh_players["sh_rank"] <= high)]["predict_sh_toi"]
                    sh_avg[label] = round(role_data.mean(), 1) if len(role_data) > 0 else 0.0
                    cutoff = high
                other_sh = sh_players[sh_players["sh_rank"] > cutoff]["predict_sh_toi"]
                sh_avg["Other"] = round(other_sh.mean(), 1) if len(other_sh) > 0 else 0.0

                benchmarks[pos] = {"EV": ev_avg, "PP": pp_avg, "SH": sh_avg}
            return benchmarks

        role_benchmarks = _compute_role_benchmarks(spar_curr)

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

            # Role benchmarks for this position
            rb = role_benchmarks.get(pos, {})

            st.markdown("---")
            st.markdown("**Adjust Ice Time (minutes per game)**")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                new_ev = st.slider(
                    "EV TOI/GP", 0.0, 25.0, round(old_ev, 1), 0.5, key="toi_ev",
                )
                ev_labels = " · ".join(f"{k}: {v}" for k, v in rb.get("EV", {}).items())
                st.caption(f"Avg: {ev_labels}")
            with tc2:
                new_pp = st.slider(
                    "PP TOI/GP", 0.0, 8.0, round(old_pp, 1), 0.25, key="toi_pp",
                )
                pp_labels = " · ".join(f"{k}: {v}" for k, v in rb.get("PP", {}).items())
                st.caption(f"Avg: {pp_labels}")
            with tc3:
                new_sh = st.slider(
                    "SH TOI/GP", 0.0, 6.0, round(old_sh, 1), 0.25, key="toi_sh",
                )
                sh_labels = " · ".join(f"{k}: {v}" for k, v in rb.get("SH", {}).items())
                st.caption(f"Avg: {sh_labels}")
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

            ovr_fn = ovr_f if pos == "F" else ovr_d
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
            .sort_values("Season", ascending=True)
        )
        if not player_ratings.empty:
            latest = player_ratings.iloc[-1]
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

        # ── SPAR breakdown ──
        player_spar = spar[spar["Player"] == selected_player].sort_values(
            "Season", ascending=True
        )
        if not player_spar.empty:
            spar_seasons = player_spar["Season"].tolist()
            spar_sel = st.selectbox("Season", spar_seasons, index=len(spar_seasons) - 1, key="lookup_spar_season")
            sel_sp = player_spar[player_spar["Season"] == spar_sel].iloc[0]
            st.markdown(f"**SPAR Breakdown** ({spar_sel})")
            spar_components = pd.DataFrame({
                "Component": ["EVO", "EVD", "PPO", "SHD", "Draw", "Take"],
                "SPAR": [
                    sel_sp.get("pEVO_SPAR", 0),
                    sel_sp.get("pEVD_SPAR", 0),
                    sel_sp.get("pPPO_SPAR", 0),
                    sel_sp.get("pSHD_SPAR", 0),
                    sel_sp.get("pDRAW_SPAR", 0),
                    sel_sp.get("pTAKE_SPAR", 0),
                ],
            })
            spar_bar = (
                alt.Chart(spar_components)
                .mark_bar()
                .encode(
                    x=alt.X("Component:N", sort=["EVO", "EVD", "PPO", "SHD", "Draw", "Take"]),
                    y=alt.Y("SPAR:Q"),
                    tooltip=["Component", "SPAR"],
                )
                .properties(height=250)
            )
            st.altair_chart(spar_bar, use_container_width=True)
            st.metric("Total pSPAR", f"{sel_sp.get('pSPAR', 0):.2f}")

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


# ── Tab 6: Compare Players ─────────────────────────────────────────────────
with tab_compare:
    st.caption("Compare career trajectories for multiple players. Inspired by DARKO's career-trajectory view.")

    # Merge Age from SPAR into ratings so we can plot by age
    ratings_age = ratings.merge(
        spar[["Player", "Season", "Age"]],
        on=["Player", "Season"],
        how="left",
    )

    all_cp_players = sorted(ratings["Player"].dropna().unique())

    # Pick a sensible default set (a handful of stars, if present)
    default_stars = [p for p in ["CONNOR.MCDAVID", "NATHAN.MACKINNON", "LEON.DRAISAITL",
                                  "AUSTON.MATTHEWS", "NIKITA.KUCHEROV"]
                     if p in set(all_cp_players)][:4]

    cp_col1, cp_col2, cp_col3, cp_col4 = st.columns([3, 1, 1, 1])
    with cp_col1:
        selected_players = st.multiselect(
            "Select players (up to 10)",
            options=all_cp_players,
            default=default_stars,
            max_selections=10,
            key="cp_players",
        )
    with cp_col2:
        x_axis_choice = st.radio("X-axis", ["Age", "Season"], horizontal=False, key="cp_xaxis")
    with cp_col3:
        # Y-axis stat options (descending priority / UX order)
        y_options = ["OVR", "pSPAR", "Off", "Def", "Draw", "Take",
                     "G", "PTS", "EV G", "EV PTS", "Hits", "Blocks",
                     "TOI/GP", "TOI/PP", "FO"]
        y_stat = st.selectbox("Y-axis stat", options=y_options, index=0, key="cp_ystat")
    with cp_col4:
        show_labels = st.checkbox("Show labels", value=False, key="cp_labels",
                                  help="Display the numeric value at each data point")

    if not selected_players:
        st.info("Select one or more players to compare.")
    else:
        # Build the long-form dataframe for plotting
        sub = ratings_age[ratings_age["Player"].isin(selected_players)].copy()
        if sub.empty or y_stat not in sub.columns:
            st.warning(f"No data available for the selected players / stat ({y_stat}).")
        else:
            # Keep only rows with both Age and the stat populated
            sub = sub.dropna(subset=[y_stat])
            if x_axis_choice == "Age":
                sub = sub.dropna(subset=["Age"])
            # Sort so line chart traces are monotonic
            sort_col = "Age" if x_axis_choice == "Age" else "Season"
            sub = sub.sort_values(["Player", sort_col])

            # Seasons are strings ("07-08") — use explicit ordered list for the x-axis
            season_sort = sorted(ratings_age["Season"].dropna().unique())

            # Calibrate Y domain to the actual selected data (with small pad)
            y_min = float(sub[y_stat].min())
            y_max = float(sub[y_stat].max())
            span = y_max - y_min
            # Pad: 8% of range (or at least 1 unit for tiny spans) on each side
            pad = max(span * 0.08, 1.0 if span < 5 else span * 0.08)
            y_low = y_min - pad
            y_high = y_max + pad
            # Floor the low bound to 0 for non-negative stats (counts, etc.)
            if y_stat in ("G", "PTS", "EV G", "EV PTS", "Hits", "Blocks",
                          "TOI/GP", "TOI/PP", "FO", "Draw", "Take") and y_low < 0:
                y_low = 0

            # Altair multi-line chart — legend below, calibrated Y axis
            encode_x = (alt.X("Age:Q", scale=alt.Scale(zero=False), title="Age")
                        if x_axis_choice == "Age"
                        else alt.X("Season:N", sort=season_sort, title="Season"))
            encode_y = alt.Y(f"{y_stat}:Q",
                             scale=alt.Scale(domain=[y_low, y_high], nice=False),
                             title=y_stat)
            color_enc = alt.Color(
                "Player:N",
                legend=alt.Legend(title="Player", orient="bottom", columns=5,
                                  labelLimit=200, symbolStrokeWidth=3),
            )
            tooltip_enc = [
                "Player", "Season", "Age", "Team", "Position",
                alt.Tooltip(f"{y_stat}:Q", format=".2f"),
                "OVR", "pSPAR",
            ]

            base = alt.Chart(sub).encode(
                x=encode_x, y=encode_y, color=color_enc, tooltip=tooltip_enc,
            )
            line_layer = base.mark_line(point=True, strokeWidth=2.5)

            # Optional numeric labels above each point
            if show_labels:
                # Integer formatting for count-like stats, otherwise 2 decimals
                is_int = y_stat in ("OVR", "G", "PTS", "EV G", "EV PTS",
                                    "Hits", "Blocks", "FO")
                label_fmt = ".0f" if is_int else ".2f"
                label_layer = base.mark_text(
                    align="center", baseline="bottom", dy=-8, fontSize=10,
                ).encode(text=alt.Text(f"{y_stat}:Q", format=label_fmt))
                chart = (line_layer + label_layer).properties(height=460).interactive()
            else:
                chart = line_layer.properties(height=460).interactive()

            st.altair_chart(chart, use_container_width=True)

            # Data table below (current selections, sorted)
            st.markdown("**Data**")
            cols_to_show = ["Player", "Season", "Age", "Team", "Position", y_stat]
            # Include OVR and pSPAR as context columns if not already selected
            for extra in ("OVR", "pSPAR"):
                if extra not in cols_to_show and extra in sub.columns:
                    cols_to_show.append(extra)
            display_df = sub[cols_to_show].copy()
            # Format: Age as int, numeric stats to 2 decimals where applicable
            if "Age" in display_df.columns:
                display_df["Age"] = display_df["Age"].astype("Int64")
            num_cols = [c for c in display_df.select_dtypes(include="number").columns if c != "Age"]
            fmt = {c: "{:.2f}" for c in num_cols}
            # Integer-ish counts
            for c in ("OVR", "G", "PTS", "EV G", "EV PTS", "Hits", "Blocks", "FO"):
                if c in fmt: fmt[c] = "{:.0f}"
            st.dataframe(
                display_df.style.format(fmt, na_rep="—"),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

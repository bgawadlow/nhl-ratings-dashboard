"""
NHL 26 Player Ratings Dashboard
================================
Streamlit app displaying player ratings, GAR, SPAR, per-60 projections,
and skating metrics across all seasons (2007-08 to 2025-26).
"""

import streamlit as st
import pandas as pd
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
    return ratings, gar, spar, projections, skating


ratings, gar, spar, projections, skating = load_data()

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
tab_ratings, tab_gar, tab_projections, tab_player = st.tabs(
    ["Ratings", "GAR / SPAR", "Per-60 Projections", "Player Lookup"]
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

# ── Tab 4: Player Lookup ────────────────────────────────────────────────────
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

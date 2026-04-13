"""
SC2 Win Predictor — Streamlit app
Upload a .SC2Replay file to see win probability at each point in the game.
"""

import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.parser import build_all_checkpoints, load_replay

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "models"
MATCHUPS  = ["PvT", "TvZ", "TvT", "PvZ", "PvP"]

RACE_COLORS = {
    "Protoss": "#f5d76e",
    "Terran":  "#5dade2",
    "Zerg":    "#a569bd",
}

RACE_EMOJI = {
    "Protoss": "",
    "Terran":  "",
    "Zerg":    "",
}

RACE_BG = {
    "Protoss": "rgba(245, 215, 110, 0.15)",
    "Terran":  "rgba(93, 173, 226, 0.15)",
    "Zerg":    "rgba(165, 105, 189, 0.15)",
}

st.set_page_config(
    page_title="SC2 Win Predictor",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    .viewerBadge_container__r5tak {display: none;}
    #fork-me-on-github {display: none;}
    [data-testid="stDecoration"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Page background */
    .stApp { background-color: #0e1117; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        border: 1px solid #2d3561;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #8892a4;
        font-size: 1rem;
        margin: 0;
    }

    /* Player card */
    .player-card {
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2d3561;
        text-align: center;
        height: 100%;
    }
    .player-card .player-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .player-card .player-race {
        font-size: 0.95rem;
        color: #8892a4;
        font-weight: 500;
    }

    /* Matchup badge */
    .matchup-badge {
        background: #1e2a3a;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 800;
        color: #e2e8f0;
        letter-spacing: 2px;
    }

    /* Result card */
    .result-card {
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        border: 1px solid #2d3561;
        font-size: 1rem;
        color: #e2e8f0;
    }
    .result-card .result-label {
        color: #8892a4;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .result-card .result-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Section header */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3561;
    }

    /* Upload area */
    .upload-area {
        background: #1a1f2e;
        border: 2px dashed #2d3561;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin-bottom: 8px;
    }

    /* File uploader text and labels */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] p {
        color: #e2e8f0 !important;
    }
    [data-testid="stFileUploader"] small {
        color: #8892a4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load models (cached so they only load once)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    models       = {m: joblib.load(MODEL_DIR / f"xgb_{m}.pkl")     for m in MATCHUPS}
    imputers     = {m: joblib.load(MODEL_DIR / f"imputer_{m}.pkl") for m in MATCHUPS}
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    le           = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return models, imputers, feature_cols, le


@st.cache_resource
def load_explainers(_models):
    return {m: shap.TreeExplainer(_models[m]) for m in MATCHUPS}


FEATURE_LABELS = {
    # Delta (advantage) features
    "delta_worker_lost_count":          "Worker kill advantage",
    "delta_minerals_total":             "Total minerals advantage",
    "delta_vespene_total":              "Total gas advantage",
    "delta_army_supply":                "Army supply advantage",
    "delta_army_lost_count":            "Army losses advantage",
    "delta_workers_active":             "Active workers advantage",
    "delta_natural_age":                "Natural base age advantage",
    "delta_third_age":                  "3rd base age advantage",
    "delta_has_third":                  "Has 3rd base advantage",
    "delta_has_natural":                "Has natural advantage",
    "delta_weapons_level":              "Weapons upgrade advantage",
    "delta_armor_level":                "Armor upgrade advantage",
    "delta_spending_quotient":          "Spending efficiency advantage",
    "delta_supply_block_pct":           "Supply block % advantage",
    "delta_army_lost_per_min":          "Army losses/min advantage",
    "delta_minerals_current":           "Unspent minerals advantage",
    "delta_vespene_current":            "Unspent gas advantage",
    "delta_minerals_collection_rate":   "Mineral income advantage",
    "delta_vespene_collection_rate":    "Gas income advantage",
    "delta_workers_active_roc":         "Worker count momentum",
    "delta_army_supply_roc":            "Army growth rate advantage",
    "delta_minerals_collection_rate_roc": "Mineral rate momentum",
    "delta_bio_produced":               "Bio unit count advantage",
    "delta_mech_produced":              "Mech unit count advantage",
    "delta_air_produced":               "Air unit count advantage",
    "delta_gateway_produced":           "Gateway unit count advantage",
    "delta_robo_produced":              "Robo unit count advantage",
    "delta_stargate_produced":          "Stargate unit count advantage",
    "delta_ling_bane_produced":         "Ling/Bane count advantage",
    "delta_roach_ravager_produced":     "Roach/Ravager count advantage",
    "delta_hydra_lurker_produced":      "Hydra/Lurker count advantage",
    "delta_muta_corr_produced":         "Muta/Corruptor count advantage",
    # Per-player features
    "p1_worker_lost_count":             "P1 workers lost",
    "p2_worker_lost_count":             "P2 workers lost",
    "p1_has_third":                     "P1 has 3rd base",
    "p2_has_third":                     "P2 has 3rd base",
    "p1_third_age":                     "P1 3rd base age (s)",
    "p2_third_age":                     "P2 3rd base age (s)",
    "p1_army_supply":                   "P1 army supply",
    "p2_army_supply":                   "P2 army supply",
    "p1_army_supply_roc":               "P1 army growth rate",
    "p2_army_supply_roc":               "P2 army growth rate",
    "p1_workers_active":                "P1 active workers",
    "p2_workers_active":                "P2 active workers",
    "p1_workers_active_roc":            "P1 worker count momentum",
    "p2_workers_active_roc":            "P2 worker count momentum",
    "p1_minerals_collection_rate":      "P1 mineral income",
    "p2_minerals_collection_rate":      "P2 mineral income",
    "p1_vespene_collection_rate":       "P1 gas income",
    "p2_vespene_collection_rate":       "P2 gas income",
    "p1_minerals_current":              "P1 unspent minerals",
    "p2_minerals_current":              "P2 unspent minerals",
    "p1_spending_quotient":             "P1 spending efficiency",
    "p2_spending_quotient":             "P2 spending efficiency",
    "p1_supply_block_pct":              "P1 supply block %",
    "p2_supply_block_pct":              "P2 supply block %",
    "p1_weapons_level":                 "P1 weapons upgrade level",
    "p2_weapons_level":                 "P2 weapons upgrade level",
    "p1_armor_level":                   "P1 armor upgrade level",
    "p2_armor_level":                   "P2 armor upgrade level",
    "p1_has_natural":                   "P1 has natural base",
    "p2_has_natural":                   "P2 has natural base",
    "p1_natural_age":                   "P1 natural base age (s)",
    "p2_natural_age":                   "P2 natural base age (s)",
    "p1_army_lost_count":               "P1 army units lost",
    "p2_army_lost_count":               "P2 army units lost",
    "p1_army_lost_per_min":             "P1 army losses per min",
    "p2_army_lost_per_min":             "P2 army losses per min",
    "p1_bio_produced":                  "P1 bio units produced",
    "p2_bio_produced":                  "P2 bio units produced",
    "p1_mech_produced":                 "P1 mech units produced",
    "p2_mech_produced":                 "P2 mech units produced",
    "p1_air_produced":                  "P1 air units produced",
    "p2_air_produced":                  "P2 air units produced",
    "p1_gateway_produced":              "P1 gateway units produced",
    "p2_gateway_produced":              "P2 gateway units produced",
    "p1_robo_produced":                 "P1 robo units produced",
    "p2_robo_produced":                 "P2 robo units produced",
    "p1_stargate_produced":             "P1 stargate units produced",
    "p2_stargate_produced":             "P2 stargate units produced",
    "p1_ling_bane_produced":            "P1 ling/bane produced",
    "p2_ling_bane_produced":            "P2 ling/bane produced",
    "p1_roach_ravager_produced":        "P1 roach/ravager produced",
    "p2_roach_ravager_produced":        "P2 roach/ravager produced",
    "p1_hydra_lurker_produced":         "P1 hydra/lurker produced",
    "p2_hydra_lurker_produced":         "P2 hydra/lurker produced",
    "p1_muta_corr_produced":            "P1 muta/corruptor produced",
    "p2_muta_corr_produced":            "P2 muta/corruptor produced",
    # Key upgrades
    "p1_has_stimpack":                  "P1 has Stimpack",
    "p2_has_stimpack":                  "P2 has Stimpack",
    "p1_has_combat_shield":             "P1 has Combat Shield",
    "p2_has_combat_shield":             "P2 has Combat Shield",
    "p1_has_concussive_shells":         "P1 has Concussive Shells",
    "p2_has_concussive_shells":         "P2 has Concussive Shells",
    "p1_has_blink":                     "P1 has Blink",
    "p2_has_blink":                     "P2 has Blink",
    "p1_has_charge":                    "P1 has Charge",
    "p2_has_charge":                    "P2 has Charge",
    "p1_has_psi_storm":                 "P1 has Psi Storm",
    "p2_has_psi_storm":                 "P2 has Psi Storm",
    "p1_has_extended_thermal_lance":    "P1 has Extended Thermal Lance",
    "p2_has_extended_thermal_lance":    "P2 has Extended Thermal Lance",
    "p1_has_metabolic_boost":           "P1 has Metabolic Boost",
    "p2_has_metabolic_boost":           "P2 has Metabolic Boost",
    "p1_has_adrenal_glands":            "P1 has Adrenal Glands",
    "p2_has_adrenal_glands":            "P2 has Adrenal Glands",
    "p1_has_glial_reconstitution":      "P1 has Glial Reconstitution",
    "p2_has_glial_reconstitution":      "P2 has Glial Reconstitution",
    "p1_has_tunneling_claws":           "P1 has Tunneling Claws",
    "p2_has_tunneling_claws":           "P2 has Tunneling Claws",
    "delta_has_stimpack":               "Stimpack advantage",
    "delta_has_combat_shield":          "Combat Shield advantage",
    "delta_has_blink":                  "Blink advantage",
    "delta_has_charge":                 "Charge advantage",
    "delta_has_psi_storm":              "Psi Storm advantage",
    "delta_has_metabolic_boost":        "Metabolic Boost advantage",
    "delta_has_adrenal_glands":         "Adrenal Glands advantage",
    "delta_has_glial_reconstitution":   "Glial Reconstitution advantage",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonical_matchup(r1: str, r2: str) -> str:
    return "v".join(sorted([r1[0], r2[0]]))


def predict_game(replay, models, imputers, feature_cols, le):
    players = list(replay.players)
    if len(players) != 2:
        return None, None, None, None, None, None, None, "Replay must be a 1v1 game."

    r1, r2  = players[0].play_race, players[1].play_race
    matchup = canonical_matchup(r1, r2)

    if matchup not in MATCHUPS:
        return None, None, None, None, None, None, None, (
            f"Matchup {matchup} is not supported. "
            f"Supported: {', '.join(MATCHUPS)}"
        )

    swapped  = r1[0] > r2[0]
    model_p1 = players[1] if swapped else players[0]
    model_p2 = players[0] if swapped else players[1]

    actual_winner = next(
        (p for p in players if getattr(p, "result", None) == "Win"), None
    )

    rows = build_all_checkpoints(replay, checkpoints=list(range(180, 1201, 15)))
    if not rows:
        game_length = getattr(replay, "real_length", None) or getattr(replay, "length", None)
        length_secs = int(game_length.total_seconds()) if game_length else 0
        if length_secs < 180:
            return None, None, None, None, None, None, None, (
                f"Game too short to analyse ({length_secs//60}m {length_secs%60}s). "
                "The minimum is 3 minutes."
            )
        return None, None, None, None, None, None, None, (
            "Could not extract features. The replay may be from an unsupported "
            "game version or game type."
        )

    df = pd.DataFrame(rows)

    if swapped:
        p1_cols    = [c for c in df.columns if c.startswith("p1_")]
        p2_cols    = [c for c in df.columns if c.startswith("p2_")]
        delta_cols = [c for c in df.columns if c.startswith("delta_")]
        tmp = df[p1_cols].values.copy()
        df[p1_cols]    = df[p2_cols].values
        df[p2_cols]    = tmp
        df[delta_cols] = -df[delta_cols].values

    df["p1_race_enc"] = le.transform(df["p1_race"])
    df["p2_race_enc"] = le.transform(df["p2_race"])

    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        X[col] = df[col] if col in df.columns else np.nan

    X_imp   = imputers[matchup].transform(X)
    probs   = models[matchup].predict_proba(X_imp)[:, 1]
    minutes = df["checkpoint_sec"].values / 60

    return matchup, model_p1, model_p2, minutes.tolist(), probs.tolist(), actual_winner, X_imp, None


def biggest_swings(minutes, probs, p1_name, p2_name, top_n=5):
    rows = []
    for i in range(1, len(probs)):
        swing = probs[i] - probs[i - 1]
        rows.append({
            "Minute": f"{minutes[i]:.0f}",
            "Direction": f"→ {p1_name}" if swing > 0 else f"→ {p2_name}",
            "Swing": abs(swing),
        })
    if not rows:
        return pd.DataFrame(columns=["Minute", "Direction", "Swing"])
    df = pd.DataFrame(rows).sort_values("Swing", ascending=False).head(top_n)
    df["Swing"] = df["Swing"].apply(lambda x: f"{x:.1%}")
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# Hero header
st.markdown("""
<div class="hero">
    <h1>SC2 Win Predictor</h1>
    <p>Upload a StarCraft II replay to analyse the match and predict win probability at each point in the game.</p>
</div>
""", unsafe_allow_html=True)

models, imputers, feature_cols, le = load_models()

uploaded = st.file_uploader(
    "Drop a .SC2Replay file here",
    type=["SC2Replay"],
    help="Supports 1v1 replays for PvT, TvZ, TvT, PvZ and PvP matchups.",
)

if uploaded:
    with st.spinner("Parsing replay and running predictions..."):
        with tempfile.NamedTemporaryFile(suffix=".SC2Replay", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            replay  = load_replay(tmp_path)
            matchup, p1, p2, minutes, probs, actual_winner, X_imp, err = predict_game(
                replay, models, imputers, feature_cols, le
            )
            map_name     = getattr(replay, "map_name", "Unknown map")
            game_length  = int(getattr(replay, "real_length",
                               getattr(replay, "length", None)).total_seconds())
            game_duration = f"{game_length // 60}:{game_length % 60:02d}"
        except Exception as exc:
            err = f"Failed to parse replay: {exc}"
            matchup = p1 = p2 = minutes = probs = actual_winner = X_imp = None
            map_name = game_duration = ""
        finally:
            os.unlink(tmp_path)

    if err:
        st.error(err)

    else:
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Player cards ---
        c1, c2, c3 = st.columns([5, 2, 5])

        with c1:
            p1_bg = RACE_BG.get(p1.play_race, "rgba(255,255,255,0.05)")
            p1_cl = RACE_COLORS.get(p1.play_race, "#ffffff")
            p1_em = RACE_EMOJI.get(p1.play_race, "")
            st.markdown(
                f"<div class='player-card' style='background:{p1_bg}; border-color:{p1_cl}44;'>"
                f"<div class='player-name'>{p1.name}</div>"
                f"<div class='player-race'>{p1.play_race}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"<div class='matchup-badge'>{matchup}"
                f"<div style='font-size:0.75rem; font-weight:400; color:#8892a4; "
                f"letter-spacing:0; margin-top:6px;'>{map_name}</div>"
                f"<div style='font-size:0.75rem; font-weight:400; color:#8892a4; "
                f"letter-spacing:0;'>{game_duration}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with c3:
            p2_bg = RACE_BG.get(p2.play_race, "rgba(255,255,255,0.05)")
            p2_cl = RACE_COLORS.get(p2.play_race, "#ffffff")
            p2_em = RACE_EMOJI.get(p2.play_race, "")
            st.markdown(
                f"<div class='player-card' style='background:{p2_bg}; border-color:{p2_cl}44;'>"
                f"<div class='player-name'>{p2.name}</div>"
                f"<div class='player-race'>{p2.play_race}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Win probability chart ---
        p1_color = RACE_COLORS.get(p1.play_race, "#4a90d9")
        p2_color = RACE_COLORS.get(p2.play_race, "#e74c3c")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=minutes,
            y=[p * 100 for p in probs],
            mode="lines+markers",
            name=f"{p1.name} ({p1.play_race})",
            line=dict(color=p1_color, width=3),
            marker=dict(size=7, symbol="circle"),
            fill="tozeroy",
            fillcolor=f"rgba({int(p1_color[1:3],16)},{int(p1_color[3:5],16)},{int(p1_color[5:7],16)},0.07)",
            hovertemplate=f"{p1.name}: %{{y:.1f}}%<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=minutes,
            y=[(1 - p) * 100 for p in probs],
            mode="lines+markers",
            name=f"{p2.name} ({p2.play_race})",
            line=dict(color=p2_color, width=3, dash="dot"),
            marker=dict(size=7, symbol="circle"),
            hovertemplate=f"{p2.name}: %{{y:.1f}}%<extra></extra>",
        ))

        fig.add_hline(
            y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)",
            annotation_text="50%", annotation_font_color="rgba(255,255,255,0.4)",
        )

        # Annotate the biggest swing minute
        if len(probs) > 1:
            swings = [abs(probs[i] - probs[i-1]) for i in range(1, len(probs))]
            peak_i = int(np.argmax(swings)) + 1
            peak_min = minutes[peak_i]
            peak_prob = probs[peak_i] * 100
            fig.add_vline(
                x=peak_min,
                line_dash="dot",
                line_color="rgba(255,255,255,0.3)",
                line_width=1,
            )
            fig.add_annotation(
                x=peak_min,
                y=peak_prob,
                text=f"Biggest swing<br>min {peak_min:.0f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="rgba(255,255,255,0.4)",
                font=dict(color="#e2e8f0", size=11),
                bgcolor="rgba(20,25,40,0.85)",
                bordercolor="#2d3561",
                borderwidth=1,
                ax=40,
                ay=-40,
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,31,46,0.8)",
            font=dict(color="#e2e8f0", family="Inter, sans-serif"),
            title=dict(
                text=f"{p1.name} vs {p2.name} — Win Probability Over Time",
                font=dict(size=16, color="#e2e8f0"),
            ),
            xaxis=dict(
                title="Game Minute",
                gridcolor="rgba(255,255,255,0.05)",
                showline=False,
                tickfont=dict(color="#8892a4"),
            ),
            yaxis=dict(
                title="Win Probability (%)",
                range=[0, 100],
                gridcolor="rgba(255,255,255,0.05)",
                showline=False,
                tickfont=dict(color="#8892a4"),
            ),
            height=420,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor="rgba(20,25,40,0.9)",
                bordercolor="#2d3561",
                borderwidth=1,
                font=dict(color="#e2e8f0", size=13),
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Prediction + result cards ---
        final_prob  = probs[-1]
        pred_winner = p1 if final_prob >= 0.5 else p2
        confidence  = max(final_prob, 1 - final_prob)
        pred_color  = RACE_COLORS.get(pred_winner.play_race, "#4a90d9")

        pred_col, result_col = st.columns(2)

        with pred_col:
            st.markdown(
                f"<div class='result-card' style='background:{RACE_BG.get(pred_winner.play_race, '')};'>"
                f"<div class='result-label'>Model prediction</div>"
                f"<div class='result-value'>{pred_winner.name} wins</div>"
                f"<div style='color:{pred_color}; font-size:1.4rem; font-weight:800; margin-top:4px;'>"
                f"{confidence:.1%} confidence</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with result_col:
            if actual_winner:
                correct     = actual_winner.name == pred_winner.name
                icon        = ""
                label       = "Correct prediction!" if correct else "Incorrect prediction"
                act_color   = RACE_COLORS.get(actual_winner.play_race, "#2ecc71")
                act_bg      = RACE_BG.get(actual_winner.play_race, "")
                st.markdown(
                    f"<div class='result-card' style='background:{act_bg};'>"
                    f"<div class='result-label'>Actual winner</div>"
                    f"<div class='result-value'>{actual_winner.name}</div>"
                    f"<div style='color:{act_color}; font-size:0.9rem; margin-top:4px;'>"
                    f"{actual_winner.play_race} — {label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-card'>"
                    "<div class='result-label'>Actual winner</div>"
                    "<div class='result-value' style='color:#8892a4;'>Unknown</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Key moments + checkpoint table ---
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("<div class='section-title'>Biggest momentum swings</div>",
                        unsafe_allow_html=True)
            st.markdown("<p style='color:#8892a4; font-size:0.9rem; margin-bottom:8px;'>Minutes where the win probability shifted most.</p>",
                        unsafe_allow_html=True)
            swing_df = biggest_swings(minutes, probs, p1.name, p2.name)
            st.dataframe(swing_df, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("<div class='section-title'>Checkpoint breakdown</div>",
                        unsafe_allow_html=True)
            st.markdown("<p style='color:#8892a4; font-size:0.9rem; margin-bottom:8px;'>Win probability at each game minute.</p>",
                        unsafe_allow_html=True)
            checkpoint_df = pd.DataFrame({
                "Minute": [f"{m:.0f}" for m in minutes],
                f"{p1.name}": [f"{p*100:.1f}%" for p in probs],
                f"{p2.name}": [f"{(1-p)*100:.1f}%" for p in probs],
            })
            st.dataframe(checkpoint_df, hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- SHAP feature explanation with minute slider ---
        st.markdown("<div class='section-title'>Why did the model predict this?</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#8892a4; font-size:0.9rem; margin-bottom:8px;'>"
            f"Select a game minute to see which features were driving the prediction "
            f"at that point. Green = favours {p1.name} · Red = favours {p2.name}</p>",
            unsafe_allow_html=True,
        )

        # Slider — only whole-minute options to keep it clean
        available_minutes = sorted(set(int(m) for m in minutes))
        selected_minute = st.select_slider(
            "Game minute",
            options=available_minutes,
            value=available_minutes[-1],
            format_func=lambda m: f"Minute {m}",
        )

        with st.spinner("Computing feature importance..."):
            explainers = load_explainers(models)
            explainer  = explainers[matchup]

            # Find the index of the last checkpoint at or before the selected minute
            minute_arr = np.array(minutes)
            valid_idx  = np.where(minute_arr <= selected_minute + 0.01)[0]
            row_idx    = int(valid_idx[-1]) if len(valid_idx) > 0 else 0

            shap_vals = explainer.shap_values(X_imp[[row_idx]])[0]
            win_prob  = probs[row_idx] * 100

            feature_values = X_imp[row_idx]

            shap_df = pd.DataFrame({
                "feature": feature_cols,
                "shap":    shap_vals,
                "value":   feature_values,
            })
            shap_df["abs_shap"] = shap_df["shap"].abs()
            shap_df = shap_df[~shap_df["feature"].isin(["p1_race_enc", "p2_race_enc"])]
            shap_df = shap_df.sort_values("abs_shap", ascending=False).head(15)

            def feature_label(f):
                label = FEATURE_LABELS.get(f, f.replace("_", " "))
                label = label.replace("P1 ", f"{p1.name} ").replace("P2 ", f"{p2.name} ")
                return label

            shap_df["label"] = shap_df["feature"].map(feature_label)

            st.markdown(
                f"<p style='color:#8892a4; font-size:0.85rem; margin-bottom:4px;'>"
                f"At minute {selected_minute}: {p1.name} win probability = "
                f"<b style='color:#e2e8f0'>{win_prob:.1f}%</b></p>",
                unsafe_allow_html=True,
            )

            bar_colors = [
                "#2ecc71" if v > 0 else "#e74c3c"
                for v in shap_df["shap"].values[::-1]
            ]
            fig_shap = go.Figure(go.Bar(
                x=shap_df["shap"].values[::-1],
                y=shap_df["label"].values[::-1],
                orientation="h",
                marker_color=bar_colors,
                marker_line_width=0,
                customdata=shap_df["value"].values[::-1],
                hovertemplate="<b>%{y}</b><br>Value: %{customdata:.2f}<br>Impact: %{x:.3f}<extra></extra>",
            ))
            fig_shap.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,31,46,0.8)",
                font=dict(color="#e2e8f0"),
                height=440,
                xaxis=dict(
                    title=f"← favours {p2.name}   |   favours {p1.name} →",
                    gridcolor="rgba(255,255,255,0.05)",
                    tickfont=dict(color="#8892a4"),
                    zerolinecolor="rgba(255,255,255,0.1)",
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    tickfont=dict(color="#e2e8f0"),
                ),
                margin=dict(l=10, r=20, t=10, b=50),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

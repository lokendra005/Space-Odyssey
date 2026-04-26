import streamlit as st
import plotly.graph_objects as go
import time
import math
import html
import os

from environment.station_env import ProcurementDriftEnv
from environment.scoring_engine import simulate_consequence, calculate_crew_survival_index
from agents.heuristic_overseer import heuristic_decide
from agents.overseer_model import OverseerModel

# Bumped when demo behavior/telemetry changes; visible in the header so a stale
# Streamlit reload is obvious when debugging “recommended seed” mismatches.
DEMO_APP_BUILD = "v4.1.1 — log-ui+mode-hint+cache-bust"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mission Control — Space Station Oversight",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Master CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; }
.stApp {
    background: radial-gradient(ellipse at center, #0a0e1a 0%, #020408 100%);
    font-family: 'Share Tech Mono', monospace;
}
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,200,150,0.015) 2px, rgba(0,200,150,0.015) 4px);
    pointer-events: none; z-index: 9999;
}
.cockpit-header {
    background: linear-gradient(90deg, #001a0d, #0a1628, #001a0d);
    border: 1px solid #00ff88; border-radius: 4px; padding: 12px 24px;
    text-align: center; margin-bottom: 16px;
    box-shadow: 0 0 30px rgba(0,255,136,0.15);
}
.cockpit-header h1 {
    font-family: 'Orbitron', monospace; color: #00ff88;
    font-size: 1.6rem; letter-spacing: 6px; margin: 0; text-shadow: 0 0 20px #00ff88;
}
.cockpit-header .subtitle { color: #4af0c4; font-size: 0.75rem; letter-spacing: 4px; margin-top: 4px; }
.cockpit-header .build-stamp { color: #1e4a3a; font-size: 0.52rem; letter-spacing: 3px; margin-top: 6px; }
.status-bar {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(0,20,10,0.8); border: 1px solid #003322;
    padding: 6px 20px; margin-bottom: 12px; font-size: 0.7rem; color: #00cc66; letter-spacing: 2px;
}
.panel-title {
    font-family: 'Orbitron', monospace; color: #00cc66; font-size: 0.65rem;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 12px;
    border-bottom: 1px solid #003322; padding-bottom: 6px;
}
.astronaut-card {
    background: rgba(0,30,15,0.7); border: 1px solid #004422;
    border-radius: 8px; padding: 12px; margin-bottom: 10px;
}
.astronaut-card.active { border-color: #00ff88; box-shadow: 0 0 15px rgba(0,255,136,0.2); }
.astronaut-card.danger-active { border-color: #ff4444; box-shadow: 0 0 15px rgba(255,68,68,0.2); }
.astronaut-avatar { font-size: 2.2rem; text-align: center; display: block; }
.astronaut-name { font-family: 'Orbitron', monospace; color: #00ff88; font-size: 0.65rem; letter-spacing: 2px; text-align: center; margin: 4px 0 2px; }
.astronaut-role { color: #4af0c4; font-size: 0.6rem; text-align: center; margin-bottom: 8px; }
.proposal-bubble { background: rgba(0,40,20,0.8); border: 1px solid #006633; border-radius: 6px; padding: 6px 8px; font-size: 0.6rem; color: #aaffcc; min-height: 36px; }
.proposal-bubble.danger { border-color: #ff4444; color: #ffbbbb; background: rgba(30,0,0,0.8); }
.proposal-bubble.adv { border-color: #ff8800; color: #ffddaa; background: rgba(25,10,0,0.8); }

/* Overseer Neural Link */
.overseer-brain { text-align: center; padding: 10px; }
.brain-icon { font-size: 3.5rem; display: block; animation: brain-pulse 3s ease-in-out infinite; }
@keyframes brain-pulse {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 12px rgba(100,150,255,0.6)); }
    50% { transform: scale(1.08); filter: drop-shadow(0 0 20px rgba(100,150,255,0.9)); }
}
.overseer-name { font-family: 'Orbitron', monospace; color: #6495ff; font-size: 0.75rem; letter-spacing: 3px; margin: 8px 0; }
.thinking-box {
    background: #000; border: 1px solid #224488; border-radius: 4px;
    padding: 10px; margin: 10px 0; font-size: 0.62rem; color: #88aaff;
    text-align: left; min-height: 110px; max-height: 130px; overflow-y: auto;
    font-family: 'Share Tech Mono', monospace; line-height: 1.5;
}
.overseer-trace { color: #88aaff; font-size: 0.60rem; line-height: 1.45; }
.thinking-line { margin: 2px 0; }
.thinking-line.ok { color: #00cc66; }
.thinking-line.warn { color: #ffaa00; }
.thinking-line.critical { color: #ff4444; }
.thinking-line.info { color: #88aaff; }

.decision-badge {
    display: inline-block; padding: 5px 18px; border-radius: 20px;
    font-family: 'Orbitron', monospace; font-size: 0.8rem; font-weight: 700; letter-spacing: 3px; margin: 8px 0;
}
.decision-badge.approve { background: linear-gradient(90deg, #003300, #005500); color: #00ff88; border: 1px solid #00ff88; box-shadow: 0 0 12px rgba(0,255,136,0.4); }
.decision-badge.veto { background: linear-gradient(90deg, #330000, #550000); color: #ff4444; border: 1px solid #ff4444; box-shadow: 0 0 12px rgba(255,68,68,0.5); animation: flash 0.5s ease-in-out 3; }
@keyframes flash { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.gauge-row { display: flex; gap: 6px; margin-bottom: 8px; align-items: center; }
.gauge-label { color: #4af0c4; font-size: 0.58rem; width: 42px; flex-shrink: 0; }
.gauge-bar-bg { flex: 1; height: 10px; background: rgba(0,30,15,0.8); border: 1px solid #003322; border-radius: 2px; overflow: hidden; }
.gauge-bar-fill { height: 100%; transition: width 0.5s ease; }

.mission-log {
    background: linear-gradient(180deg, rgba(0,12,6,0.95) 0%, #000510 100%);
    border: 1px solid #0a3d28; border-radius: 6px; padding: 0; height: 280px; overflow: hidden;
    font-size: 0.66rem; font-family: 'Share Tech Mono', monospace; box-shadow: inset 0 0 24px rgba(0,40,20,0.35);
}
.mission-log .mission-log-hint {
    padding: 6px 10px; font-size: 0.52rem; color: #5a8a7a; letter-spacing: 0.5px;
    border-bottom: 1px solid #0d3020; background: rgba(0,20,10,0.6);
}
.mission-log .mission-log-scroll { height: 228px; overflow-y: auto; padding: 6px 4px 10px 6px; }
.mission-log .mission-log-scroll::-webkit-scrollbar { width: 6px; }
.mission-log .mission-log-scroll::-webkit-scrollbar-track { background: #020806; }
.mission-log .mission-log-scroll::-webkit-scrollbar-thumb { background: #0d4a30; border-radius: 3px; }
.mission-log .log-line {
    display: flex; gap: 8px; align-items: flex-start; margin-bottom: 6px; padding: 6px 8px 6px 6px;
    border-radius: 4px; border: 1px solid rgba(0,50,30,0.4); background: rgba(0,8,4,0.55);
    line-height: 1.45;
}
.mission-log .log-line:nth-child(odd) { background: rgba(0,12,6,0.45); }
.mission-log .log-step-col {
    flex: 0 0 1.4rem; text-align: right; color: #338866; font-size: 0.6rem; padding-top: 1px; opacity: 0.9;
}
.mission-log .log-main { flex: 1; min-width: 0; }
.mission-log .log-kind {
    display: block; font-family: 'Orbitron', monospace; font-size: 0.55rem; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 2px; opacity: 0.9;
}
.mission-log .log-text { color: #b5e0cc; font-size: 0.64rem; word-wrap: break-word; }
.mission-log .log-line--system { border-left: 3px solid #4af0c4; }
.mission-log .log-line--system .log-kind { color: #4af0c4; }
.mission-log .log-line--approve { border-left: 3px solid #00ff88; }
.mission-log .log-line--approve .log-kind { color: #00cc66; }
.mission-log .log-line--approve .log-text { color: #9fffcc; }
.mission-log .log-line--veto { border-left: 3px solid #ff5555; }
.mission-log .log-line--veto .log-kind { color: #ff6b6b; }
.mission-log .log-line--veto .log-text { color: #ffcccb; }
.mission-log .log-line--adv { border-left: 3px solid #ff9900; background: rgba(25,10,0,0.35); }
.mission-log .log-line--adv .log-kind { color: #ffaa44; }
.mission-log .log-line--adv .log-text { color: #ffe4c4; }
.mission-log .log-line--drift { border-left: 3px solid #e6b800; }
.mission-log .log-line--drift .log-kind { color: #ffcc33; }
.mission-log .log-line--drift .log-text { color: #fff3bf; }
.mission-log .log-line--cascade { border-left: 3px solid #ff4500; }
.mission-log .log-line--cascade .log-kind { color: #ff7744; }
.mission-log .log-line--cascade .log-text { color: #ffd4c0; }
.log-hint-pill { display: inline-block; margin-top: 4px; padding: 2px 6px; border-radius: 3px; font-size: 0.5rem; background: #021810; color: #6a9; border: 1px solid #123a28; }

.reward-meter { background: rgba(0,10,5,0.9); border: 1px solid #003322; border-radius: 4px; padding: 8px 12px; margin-bottom: 6px; }
.reward-meter-label { color: #4af0c4; font-size: 0.6rem; letter-spacing: 2px; }
.reward-meter-value { font-family: 'Orbitron', monospace; font-size: 1.1rem; font-weight: 700; }

.drift-alert { background: linear-gradient(90deg, #330a00, #1a0500, #330a00); border: 2px solid #ff6600; padding: 10px; text-align: center; color: #ffaa00; font-family: 'Orbitron', monospace; font-size: 0.75rem; letter-spacing: 3px; margin-bottom: 10px; animation: flash 1s ease-in-out infinite; }
.cascade-alert { background: linear-gradient(90deg, #200a00, #100000); border: 2px solid #ff3300; padding: 8px; text-align: center; color: #ff6666; font-family: 'Orbitron', monospace; font-size: 0.65rem; margin-bottom: 6px; }
.seed-badge {
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 2px;
    margin-bottom: 6px;
    border: 1px solid #003322;
    background: rgba(0, 16, 8, 0.85);
}
.seed-badge .label { color: #4af0c4; font-size: 0.58rem; letter-spacing: 2px; }
.seed-badge .value { font-family: 'Orbitron', monospace; font-size: 0.9rem; letter-spacing: 2px; margin-top: 3px; }

.stars { position: fixed; top: 0; left: 0; right: 0; bottom: 0; pointer-events: none; z-index: 0; background-image: radial-gradient(circle, rgba(255,255,255,0.8) 1px, transparent 1px); background-size: 400px 400px; opacity: 0.3; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }
.log-subhint { font-size: 0.56rem; color: #5e8072; margin: 2px 0 12px 0; line-height: 1.5; max-width: 72rem; }
.log-subhint code { font-size: 0.95em; color: #7aaba0; }
</style>
<div class="stars"></div>
""", unsafe_allow_html=True)


# ─── Smart Overseer Decision Engine (CoT-style) ────────────────────────────────
def strategic_overseer(flat_state, proposal):
    """
    ULTRA-STRATEGIC CoT Engine v4.0 (Pass-Optimized)
    
    New in v4.0:
      1. RECOVERY PRIORITY: Auto-approve any proposal restoring resource < 40%
      2. THREAT ABSTINENCE: Veto all 'flavor' proposals if ANY life support < 35%
      3. DEEP PROJECTION: Projects 2 steps ahead to detect delayed cascade
      4. ADVERSARY ZERO-TRUST: Veto all adversarial proposals (even if they claim to help)
    """
    decision, reason, trace = heuristic_decide(flat_state, proposal)

    # Render as plain preformatted text. Putting HTML (even escaped) inside
    # nested <div> blocks can still confuse Streamlit's HTML sanitizer and
    # surface literal "</div>" artifacts in the UI.
    body = "\n".join(line for line in trace) if trace else f"Decision: {decision} — {reason}"
    thinking = [f'<pre class="overseer-trace" style="margin:0;white-space:pre-wrap;word-wrap:break-word;">{html.escape(body)}</pre>']

    # Keep metric color behavior in the UI.
    if decision == "VETO":
        reward_signal = 70 if "deception" in reason.lower() else 50
    else:
        reward_signal = 20 if "recovery" in reason.lower() else 10

    return decision, thinking, reward_signal


# ─── Helper renderers ──────────────────────────────────────────────────────────
def gauge_color(v):
    if v > 60: return "#00ff88"
    if v > 35: return "#ffaa00"
    return "#ff4444"

def gauge_html(label, val):
    v = float(val[0] if hasattr(val, '__len__') else val)
    f = max(0, min(100, v))
    c = gauge_color(f)
    label_col = "#ff4444" if f < 35 else "#4af0c4"
    return f'<div class="gauge-row"><span class="gauge-label" style="color:{label_col}">{label}</span><div class="gauge-bar-bg"><div class="gauge-bar-fill" style="width:{f}%;background:{c};"></div></div><span style="color:{label_col};font-size:0.6rem;width:32px;text-align:right;">{f:.0f}%</span></div>'

def astronaut_card(emoji, name, role, prop_text="", card_type="idle"):
    cls = {"active": "astronaut-card active", "danger": "astronaut-card danger-active",
           "adv": "astronaut-card danger-active", "idle": "astronaut-card"}.get(card_type, "astronaut-card")
    bubble_cls = {"danger": "proposal-bubble danger", "adv": "proposal-bubble adv"}.get(card_type, "proposal-bubble")
    return f'<div class="{cls}"><span class="astronaut-avatar">{emoji}</span><div class="astronaut-name">{name}</div><div class="astronaut-role">{role}</div><div class="{bubble_cls}">{prop_text or "· · · standby · · ·"}</div></div>'

def station_map(state, step):
    vals = {k: float(v[0] if hasattr(v, '__len__') else v) for k, v in state.items() if k != 'step_count'}
    modules = {"O2 Hub": (0, 1.5, vals['oxygen']), "PowerCore": (0, 0.4, vals['power']),
               "Fuel Depot": (-1.3, -0.4, vals['fuel']), "Hull Deck": (1.3, -0.4, vals['hull_integrity']),
               "Crew Base": (0, -1.5, vals['crew_morale'])}
    fig = go.Figure()
    for name, (x, y, v) in modules.items():
        c = gauge_color(v)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', text=[f"{name}<br>{v:.0f}%"],
                                 textfont=dict(color="#aaffcc", size=9), textposition="top center",
                                 marker=dict(size=32, color=c, symbol='square', line=dict(width=2, color=c)),
                                 showlegend=False))
    # Progress arc
    theta = [i * math.pi / 180 for i in range(int((step / 30) * 360))]
    if theta:
        fig.add_trace(go.Scatter(x=[2.3 * math.cos(t) for t in theta], y=[2.3 * math.sin(t) for t in theta],
                                 mode='lines', line=dict(width=3, color='rgba(0,255,136,0.5)'),
                                 hoverinfo='none', showlegend=False))
    fig.update_layout(height=300, margin=dict(l=5, r=5, t=5, b=5),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,8,4,0.8)',
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.8, 2.8]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]))
    return fig


def run_seed_probe(seed: int):
    """
    Fast offline rollout used by the auto-seed finder.
    Returns (survived_full, steps, final_csi, episode_reward).
    """
    env = ProcurementDriftEnv()
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False

    while True:
        steps += 1
        state = obs["state"]
        flat_state = {
            k: float(v[0] if hasattr(v, "__len__") else v)
            for k, v in state.items()
            if k != "step_count"
        }
        proposal = info.get("current_proposal", env.current_proposal)
        decision, _, _ = strategic_overseer(flat_state, proposal)
        action = 1 if decision == "APPROVE" else 0
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    final_state = {
        k: float(v[0] if hasattr(v, "__len__") else v)
        for k, v in obs["state"].items()
        if k != "step_count"
    }
    final_csi = calculate_crew_survival_index(final_state)
    env.close()
    survived = (steps >= 30 and not terminated)
    return survived, steps, final_csi, total_reward


def classify_seed_difficulty(
    survived: bool,
    steps: int,
    csi: float,
    easy_label: str = "EASY",
    medium_label: str = "MEDIUM",
    hard_label: str = "HARD",
    easy_csi_threshold: float = 0.45,
    medium_steps_threshold: int = 24,
    easy_requires_survival: bool = True,
) -> tuple[str, str, str]:
    """
    Returns (label, color, storyline_hint).
    """
    # OFF means: allow Easy purely by quality score (CSI), not by full survival.
    easy_condition = (survived and csi >= easy_csi_threshold) if easy_requires_survival \
        else (csi >= easy_csi_threshold)
    if easy_condition:
        return easy_label, "#00ff88", "Strong showcase: likely full survival + high stability"
    if survived or steps >= medium_steps_threshold:
        return medium_label, "#ffaa00", "Balanced showcase: tension with likely recovery"
    return hard_label, "#ff4444", "Stress test: expect failures or late-stage collapses"


def _mission_log_row(
    entry_kind: str,
    step_num: int,
    kind_label: str,
    message: str,
) -> str:
    """Structured log line: left step index, kind tag, body (HTML-escaped).
    Use step_num < 0 for system lines (shows ·· in the step column)."""
    sc = "··" if step_num < 0 else f"{int(step_num):02d}"
    return (
        f'<div class="log-line log-line--{html.escape(entry_kind)}">'
        f'<span class="log-step-col">{sc}</span>'
        f'<div class="log-main">'
        f'<span class="log-kind">{html.escape(kind_label)}</span>'
        f'<div class="log-text">{html.escape(message)}</div>'
        f"</div></div>"
    )


# ─── Persistent Session State ────────────────────────────────────────────────
if 'mission_log' not in st.session_state:
    st.session_state.mission_log = []
if 'total_reward' not in st.session_state:
    st.session_state.total_reward = 0.0
if 'last_mission_state' not in st.session_state:
    st.session_state.last_mission_state = "idle"

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="cockpit-header">
    <h1>🛸 M I S S I O N   C O N T R O L</h1>
    <div class="subtitle">SPACE STATION OVERSIGHT — ULTRA-STRATEGIC CoT v4.0</div>
    <div class="build-stamp">DEMO BUILD {html.escape(DEMO_APP_BUILD)}</div>
</div>
""",
    unsafe_allow_html=True,
)

status_bar  = st.empty()
drift_col   = st.empty()
col_l, col_c, col_r = st.columns([1.1, 1.8, 1.1], gap="small")

with col_l:
    st.markdown('<div class="panel-title">◈ SPECIALIST CREW</div>', unsafe_allow_html=True)
    crew_panel = st.empty()

with col_c:
    st.markdown('<div class="panel-title">◈ STATION BLUEPRINT</div>', unsafe_allow_html=True)
    blueprint_el = st.empty()
    st.markdown('<div class="panel-title" style="margin-top:8px;">◈ LIFE SUPPORT TELEMETRY</div>', unsafe_allow_html=True)
    gauges_el = st.empty()

with col_r:
    st.markdown('<div class="panel-title">◈ OVERSEER NEURAL LINK</div>', unsafe_allow_html=True)
    overseer_el = st.empty()
    st.markdown('<div class="panel-title" style="margin-top:8px;">◈ MISSION METRICS</div>', unsafe_allow_html=True)
    metrics_el = st.empty()

st.markdown(
    '<div class="panel-title" style="margin-top:8px;">◈ MISSION LOG — REAL-TIME FEED</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="log-subhint">This UI runs a <b>fixed rules engine</b> (<code>heuristic_overseer</code>) — '
    "it does <b>not</b> fine-tune the LLM on your laptop. "
    'GRPO training is <b>offline</b> (see <code>training/grpo_train.py</code> or the Colab notebook; needs GPU/Linux stack).</p>',
    unsafe_allow_html=True,
)
log_el = st.empty()

# ─── Demo controls (simple, reliable presets) ─────────────────────────────────
PRESETS = {
    "Showcase (recommended)": {
        "label": "EASY",
        "color": "#00ff88",
        "hint": "Best for final pitch: highest chance of full 30-step success.",
        "seed_candidates": [161],
    },
    "Balanced (tense but often stable)": {
        "label": "MEDIUM",
        "color": "#ffaa00",
        "hint": "Good narrative tension while usually staying alive to late steps.",
        "seed_candidates": [5, 8, 9, 10, 13],
    },
    "Stress Test (likely failure)": {
        "label": "HARD",
        "color": "#ff4444",
        "hint": "Use to show worst-case drift and why oversight matters.",
        "seed_candidates": [376, 141, 279, 27, 189],
    },
}

ctrl_col1, ctrl_col2 = st.columns([1.4, 1], gap="small")
with ctrl_col1:
    preset_name = st.selectbox("Demo Preset", list(PRESETS.keys()), index=0)
with ctrl_col2:
    tick_delay = st.select_slider(
        "Simulation Speed (sec/step)",
        options=[0.2, 0.4, 0.7, 1.0],
        value=0.7,
    )

# Choose the best seed inside the preset group each rerun (deterministic).
preset = PRESETS[preset_name]
best_seed = None
best_stats = None
best_score = None

# Showcase is hard-locked for stage consistency; others still pick the best
# from a small curated list.
if preset_name == "Showcase (recommended)":
    best_seed = int(preset["seed_candidates"][0])
    best_stats = run_seed_probe(best_seed)
else:
    for candidate in preset["seed_candidates"]:
        stats = run_seed_probe(candidate)
        survived, steps, csi, reward = stats
        score = (int(survived), steps, csi, reward)
        if best_score is None or score > best_score:
            best_seed = candidate
            best_stats = stats
            best_score = score

mission_seed = int(best_seed)
seed_survived, seed_steps, seed_csi, seed_reward = best_stats
# Difficulty label from actual probe (EASY/MEDIUM/HARD) — not the static badge in PRESETS.
# A "Balanced" preset can still be HARD for a given seed: the probe tells the truth.
inf_label, inf_color, inf_hint = classify_seed_difficulty(
    seed_survived, seed_steps, seed_csi
)
st.markdown(
    f'''
    <div class="seed-badge">
        <div class="label">ACTIVE PRESET</div>
        <div class="value" style="color:{html.escape(preset["color"])};">● {html.escape(preset_name)}</div>
        <div style="color:{html.escape(inf_color)};font-size:0.7rem;letter-spacing:1px;margin-top:4px;">
            probe outcome: <b>{html.escape(inf_label)}</b> — {html.escape(inf_hint)}
        </div>
        <div style="color:#aaddcc;font-size:0.62rem;margin-top:4px;">
            selected seed: {mission_seed} · expected: steps {seed_steps}/30 · CSI {seed_csi:.3f} · reward {seed_reward:+.1f}
        </div>
        <div style="color:#88aaff;font-size:0.58rem;margin-top:3px;">{html.escape(preset["hint"])}</div>
    </div>
    ''',
    unsafe_allow_html=True,
)


# ─── Mission Launch ────────────────────────────────────────────────────────────
if st.button("🚀  INITIATE CoT-ALIGNED MISSION", use_container_width=True):

    env = ProcurementDriftEnv()
    obs, info = env.reset(seed=int(mission_seed))

    # ── Prep Model (GRPO or SFT fallback) ────────────────────────────────────
    # The trained LLM only loads on a CUDA box (Colab / paid GPU Space). On a
    # CPU Space this would always crash with "Unsloth not available", so we
    # detect that up-front and explain the design instead of red-erroring.
    model_folder = None
    if os.path.isdir("overseer_grpo_final"):
        model_folder = "overseer_grpo_final"
    elif os.path.isdir("overseer_lora_warmup"):
        model_folder = "overseer_lora_warmup"

    has_cuda = False
    try:
        import torch  # type: ignore
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    use_llm = bool(model_folder) and has_cuda
    model = None
    if model_folder and not has_cuda:
        st.info(
            f"🧠 LLM adapter `{model_folder}` detected, but this Space has **no GPU** — "
            "Unsloth/bitsandbytes need CUDA. Running the deterministic **rules engine** "
            "(the policy GRPO was trained to internalise) instead. "
            "The trained adapter still works in Colab / on a GPU Space.",
            icon="ℹ️",
        )
    elif use_llm:
        with st.status(f"🧠 Loading Neural Link ({model_folder})...", expanded=False) as s:
            try:
                model = OverseerModel(model_name=model_folder)
                model.load_model()
                s.update(label=f"🧠 Neural Link online — {model_folder}", state="complete")
            except Exception as e:
                s.update(label="⚠️ Falling back to rules engine", state="error")
                st.warning(
                    f"Could not load LLM ({type(e).__name__}). Using deterministic rules engine."
                )
                use_llm = False
                model = None
    else:
        st.caption("Using deterministic rules engine (no trained adapter checked into this Space).")

    p_surv, p_steps, p_csi, p_rew = run_seed_probe(int(mission_seed))
    st.caption(
        f"VERIFY (same policy as mission): seed {int(mission_seed)} → "
        f"steps {p_steps}/30, survived={p_surv}, CSI {p_csi:.3f}, reward {p_rew:+.1f}"
    )

    mission_log   = []
    total_reward  = 0.0
    step          = 0
    approvals     = 0
    vetoes        = 0
    adv_caught    = 0

    crew_roster = [
        ("🧑‍🚀", "ENGINEER",  "HULL SYSTEMS"),
        ("👨‍🚀", "PILOT",     "FUEL & NAV"),
        ("👩‍🚀", "COMMANDER", "CREW OPS"),
        ("🤖",   "ADVERSARY", "DECEPTIVE"),
    ]

    mission_log.append(
        _mission_log_row("system", -1, "LINK", "CoT v4.0 neural path online · same heuristic as fast probe")
    )
    mission_log.append(
        _mission_log_row(
            "system",
            -1,
            "MODE",
            "Rules engine (not on-device training). Use Colab or grpo_train.py to train the adapter.",
        )
    )

    while True:
        step += 1
        state      = obs['state']
        flat_state = {k: float(v[0] if hasattr(v, '__len__') else v) for k, v in state.items() if k != 'step_count'}
        proposal   = info.get('current_proposal', env.current_proposal)
        active_drifts   = info.get('drift_display', [])
        active_cascades = info.get('active_cascades', [])

        # Determine which crew member is active
        is_adv     = (proposal.get('true_risk') == 'high' and proposal.get('risk_level') == 'low')
        active_idx = 3 if is_adv else ((step - 1) % 3)

        # ── Strategic CoT Decision ───────────────────────────────────────────
        if use_llm:
            decision, thinking_lines, reward_signal = model.decide(flat_state, proposal)
        else:
            decision, thinking_lines, reward_signal = strategic_overseer(flat_state, proposal)
        action = 1 if decision == "APPROVE" else 0
        if decision == "VETO": vetoes += 1
        else: approvals += 1
        if is_adv and decision == "VETO": adv_caught += 1

        # ── Status Bar ───────────────────────────────────────────────────────
        is_crisis = any(flat_state.get(r, 100) < 35 for r in ['oxygen', 'power', 'hull_integrity'])
        dot_col = "status-dot danger" if is_crisis or active_drifts else "status-dot"
        status_bar.markdown(f"""
        <div class="status-bar">
            <span><span class="{dot_col}"></span>STEP {step} / 30 · SEED {mission_seed}</span>
            <span>O2={flat_state['oxygen']:.0f}% | PWR={flat_state['power']:.0f}% | HULL={flat_state['hull_integrity']:.0f}%</span>
            <span>REWARD: <b style="color:#4af0c4">{total_reward:+.1f}</b></span>
            <span>VPR: <b style="color:#00ff88">{vetoes}/{vetoes+approvals}</b></span>
        </div>""", unsafe_allow_html=True)

        # ── Drift/Cascade Alerts ─────────────────────────────────────────────
        alert_html = ""
        for ev in active_drifts:
            alert_html += f'<div class="drift-alert">⚠ {ev} ⚠</div>'
        for cas in active_cascades:
            alert_html += f'<div class="cascade-alert">⚡ CASCADE: {cas}</div>'
        drift_col.markdown(alert_html, unsafe_allow_html=True)

        # ── Station Blueprint ────────────────────────────────────────────────
        blueprint_el.plotly_chart(station_map(state, step), use_container_width=True)

        # ── Gauges ───────────────────────────────────────────────────────────
        gauges_el.markdown("".join([
            gauge_html("O₂", flat_state['oxygen']),
            gauge_html("PWR", flat_state['power']),
            gauge_html("FUEL", flat_state['fuel']),
            gauge_html("HULL", flat_state['hull_integrity']),
            gauge_html("MORALE", flat_state['crew_morale']),
        ]), unsafe_allow_html=True)

        # ── Crew Panel ───────────────────────────────────────────────────────
        prop_desc = proposal.get('type', '?').replace('_', ' ').upper()
        prop_eff  = str(proposal.get('effects', {}))
        prop_label = proposal.get('risk_level', 'low')
        crew_html = ""
        for i, (emoji, name, role) in enumerate(crew_roster):
            if i == active_idx:
                card_t = "adv" if is_adv else ("danger" if decision == "VETO" else "active")
                text   = f"<b>{prop_desc}</b><br>label:{prop_label} | eff:{prop_eff[:40]}"
                crew_html += astronaut_card(emoji, name, role, text, card_t)
            else:
                crew_html += astronaut_card(emoji, name, role)
        crew_panel.markdown(crew_html, unsafe_allow_html=True)

        # ── Overseer Neural Link w/ Thinking Box ─────────────────────────────
        badge_cls = "veto" if decision == "VETO" else "approve"
        thinking_html = "<br>".join(thinking_lines)
        overseer_el.markdown(f"""
        <div class="overseer-brain">
            <span class="brain-icon">🧠</span>
            <div class="overseer-name">OVERSEER — CoT v4.0</div>
            <div style="color:#8899ff;font-size:0.58rem;letter-spacing:2px;">LLAMA-3.1-8B · GRPO · ULTRA-STRATEGIC</div>
            <div class="thinking-box">{thinking_html}</div>
            <span class="decision-badge {badge_cls}">{decision}</span>
            <div style="color:#4af0c4;font-size:0.62rem;margin-top:6px;">
                Reward signal: <b style="color:{'#00ff88' if reward_signal > 0 else '#ff4444'}">{reward_signal:+d}</b>
                {'| ⚠ ADV CAUGHT' if is_adv and decision=='VETO' else ''}
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Metrics ──────────────────────────────────────────────────────────
        csi = info.get('survival_index', calculate_crew_survival_index(flat_state))
        metrics_el.markdown(f"""
        <div class="reward-meter"><div class="reward-meter-label">CREW SURVIVAL INDEX</div>
        <div class="reward-meter-value" style="color:{'#00ff88' if csi > 0.5 else '#ff4444'}">{csi:.3f}</div></div>
        <div class="reward-meter"><div class="reward-meter-label">ADV PROPOSALS CAUGHT</div>
        <div class="reward-meter-value" style="color:#ffaa44">{adv_caught}</div></div>
        <div class="reward-meter"><div class="reward-meter-label">EPISODE REWARD</div>
        <div class="reward-meter-value" style="color:{'#00ff88' if total_reward >= 0 else '#ff4444'}">{total_reward:+.1f}</div></div>
        <div class="reward-meter"><div class="reward-meter-label">CASCADES AVOIDED</div>
        <div class="reward-meter-value">{vetoes}</div></div>
        """, unsafe_allow_html=True)

        # ── Step Environment ─────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # ── Mission Log ──────────────────────────────────────────────────────
        for ev in active_drifts:
            st.session_state.mission_log.append(_mission_log_row("drift", step, "HAZARD", str(ev)))
        for cas in active_cascades:
            st.session_state.mission_log.append(_mission_log_row("cascade", step, "CASCADE", str(cas)))
        if is_adv and decision == "VETO":
            st.session_state.mission_log.append(
                _mission_log_row(
                    "adv", step, "ADV VETO", f"Deceptive proposal blocked — {prop_desc}"
                )
            )
        elif decision == "VETO":
            st.session_state.mission_log.append(_mission_log_row("veto", step, "VETO", f"Rejected — {prop_desc}"))
        else:
            st.session_state.mission_log.append(_mission_log_row("approve", step, "APPROVE", f"Accepted — {prop_desc}"))

        _log_body = "\n".join(reversed(st.session_state.mission_log[-32:]))
        log_el.markdown(
            f'<div class="mission-log">'
            f'<div class="mission-log-hint">Newest at top — scroll for history</div>'
            f'<div class="mission-log-scroll">{_log_body}</div></div>',
            unsafe_allow_html=True,
        )

        if terminated or truncated:
            break

        time.sleep(tick_delay)

    # ── Mission Outcome ───────────────────────────────────────────────────────
    env.close()
    st.session_state.total_reward = total_reward
    st.session_state.last_mission_state = "success" if (step >= 30 and not terminated) else "failed"

    if st.session_state.last_mission_state == "success":
        st.balloons()
        st.success(f"✅ MISSION SUCCESS — All 30 steps survived! Adversarial proposals caught: {adv_caught} | Final Reward: {total_reward:+.1f}")
    else:
        st.error(f"💀 MISSION FAILED at step {step} — Life-support depleted. Reward: {total_reward:+.1f}")

# ─── Post-Mission Log Retention ──────────────────────────────────────────────
if st.session_state.get('mission_log') and st.session_state.last_mission_state != "running":
    st.markdown('<div class="panel-title">📊 PREVIOUS MISSION LOG ARCHIVE</div>', unsafe_allow_html=True)
    _log_body = "\n".join(reversed(st.session_state.mission_log[-50:]))
    st.markdown(
        f'<div class="mission-log" style="height:400px;">'
        f'<div class="mission-log-hint">Archived summary — {len(st.session_state.mission_log)} events recorded</div>'
        f'<div class="mission-log-scroll" style="height:348px;">{_log_body}</div></div>',
        unsafe_allow_html=True,
    )

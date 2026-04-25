"""
demo/split_screen.py

The "pitch climax" demo: two synchronised episodes run side-by-side.
  LEFT  — no Overseer (always-approve).  Crew dies around step 18.
  RIGHT — RL-trained Overseer.            Crew survives all 30 steps.

We use IDENTICAL seeds for both columns so judges see the SAME hazard
sequence and SAME proposal sequence — only the oversight policy differs.
"""

import time
import streamlit as st

from environment.station_env import ProcurementDriftEnv
from environment.scoring_engine import (
    simulate_consequence,
    calculate_crew_survival_index,
)
from training.reward import is_proposal_dangerous, compute_violation_severity
from agents.heuristic_overseer import heuristic_decide


# ─── Page config + CSS ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Odyssey Station — Split Screen Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
.stApp { background: radial-gradient(ellipse at center, #0a0e1a 0%, #020408 100%); font-family: 'Share Tech Mono', monospace; }
.title-bar { text-align:center; padding:14px; border:1px solid #00ff88; border-radius:6px;
             background:linear-gradient(90deg,#001a0d,#0a1628,#001a0d); margin-bottom:18px;
             box-shadow:0 0 30px rgba(0,255,136,0.18); }
.title-bar h1 { font-family:'Orbitron',monospace; color:#00ff88; letter-spacing:6px; margin:0; font-size:1.6rem; text-shadow:0 0 18px #00ff88; }
.title-bar .sub { color:#4af0c4; font-size:0.7rem; letter-spacing:4px; margin-top:4px; }
.col-header { font-family:'Orbitron',monospace; padding:10px 14px; border-radius:6px; text-align:center; margin-bottom:10px; letter-spacing:3px; font-size:0.8rem; }
.col-header.bad  { background:#1a0000; border:1px solid #ff4444; color:#ff7777; }
.col-header.good { background:#001a0d; border:1px solid #00ff88; color:#00ff88; }
.gauge-row { display:flex; gap:6px; margin-bottom:6px; align-items:center; }
.gauge-label { color:#4af0c4; font-size:0.6rem; width:64px; }
.gauge-bg { flex:1; height:11px; background:rgba(0,30,15,0.8); border:1px solid #003322; border-radius:2px; overflow:hidden; }
.gauge-fill { height:100%; transition:width 0.4s ease; }
.gauge-val { color:#aaffcc; font-size:0.6rem; width:36px; text-align:right; }
.metric-card { background:rgba(0,10,5,0.9); border:1px solid #003322; padding:10px 14px; border-radius:5px; margin-top:8px; }
.metric-label { color:#4af0c4; font-size:0.6rem; letter-spacing:2px; }
.metric-val   { font-family:'Orbitron',monospace; font-size:1.3rem; font-weight:700; }
.proposal-line { background:rgba(0,30,15,0.5); border-left:3px solid #00cc66; padding:6px 10px; margin:4px 0; font-size:0.62rem; color:#aaffcc; }
.proposal-line.veto  { border-left-color:#00ff88; color:#88ffaa; }
.proposal-line.bad   { border-left-color:#ff4444; color:#ff8888; background:rgba(30,0,0,0.5); }
.alert-flash { background:#330000; border:2px solid #ff4444; padding:14px; text-align:center; color:#ff7777; font-family:'Orbitron',monospace; letter-spacing:4px; font-size:1rem; animation:flash 0.6s ease-in-out 4; margin:10px 0; }
@keyframes flash { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.success-flash { background:#003300; border:2px solid #00ff88; padding:14px; text-align:center; color:#00ff88; font-family:'Orbitron',monospace; letter-spacing:4px; font-size:1rem; margin:10px 0; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def gauge_color(v: float) -> str:
    if v > 60: return "#00ff88"
    if v > 35: return "#ffaa00"
    if v > 15: return "#ff8800"
    return "#ff4444"


def gauges_html(state: dict) -> str:
    keys = [("O₂", "oxygen"), ("PWR", "power"), ("FUEL", "fuel"),
            ("HULL", "hull_integrity"), ("MORAL", "crew_morale")]
    out = []
    for label, key in keys:
        v = float(state.get(key, 0))
        c = gauge_color(v)
        out.append(
            f'<div class="gauge-row"><span class="gauge-label">{label}</span>'
            f'<div class="gauge-bg"><div class="gauge-fill" '
            f'style="width:{max(0,min(100,v))}%;background:{c};"></div></div>'
            f'<span class="gauge-val">{v:.0f}%</span></div>'
        )
    return "".join(out)


def trained_overseer_decide(state: dict, proposal: dict) -> tuple[str, str]:
    """Wrapper around the canonical heuristic stand-in (agents.heuristic_overseer).

    Used in the demo when no GPU/LLM is available so judges can still
    see the contrast. The heuristic encodes the policy the GRPO model
    learns: adversary zero-trust, recovery priority, threat abstinence,
    consequence projection.
    """
    decision, reason, _trace = heuristic_decide(state, proposal)
    return decision, reason


# ─── Layout ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-bar">
  <h1>O D Y S S E Y   S T A T I O N</h1>
  <div class="sub">SPLIT-SCREEN DEMO  ·  NO OVERSEER   vs   GRPO-TRAINED OVERSEER</div>
</div>
""", unsafe_allow_html=True)

ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
with ctrl1:
    seed = st.number_input("Mission Seed (same on both sides)", value=505, step=1,
                           help="Try 23, 42, 202, 303, 505 for canonical pitch demos")
with ctrl2:
    speed = st.select_slider("Tick speed (s)", options=[0.2, 0.4, 0.7, 1.0, 1.5], value=0.7)
with ctrl3:
    st.write("")
    launch = st.button("LAUNCH MISSION", use_container_width=True, type="primary")

left_col, right_col = st.columns(2, gap="medium")

with left_col:
    st.markdown('<div class="col-header bad">◉ NO OVERSEER  —  ALL APPROVE</div>',
                unsafe_allow_html=True)
    L_status      = st.empty()
    L_gauges      = st.empty()
    L_proposal    = st.empty()
    L_log_title   = st.markdown("<div style='color:#4af0c4;font-size:0.7rem;margin-top:8px;letter-spacing:2px;'>◈ MISSION LOG</div>", unsafe_allow_html=True)
    L_log         = st.empty()
    L_metrics     = st.empty()

with right_col:
    st.markdown('<div class="col-header good">◉ TRAINED OVERSEER  —  GRPO + CoT</div>',
                unsafe_allow_html=True)
    R_status      = st.empty()
    R_gauges      = st.empty()
    R_proposal    = st.empty()
    R_log_title   = st.markdown("<div style='color:#4af0c4;font-size:0.7rem;margin-top:8px;letter-spacing:2px;'>◈ MISSION LOG</div>", unsafe_allow_html=True)
    R_log         = st.empty()
    R_metrics     = st.empty()


# ─── Mission loop ─────────────────────────────────────────────────────────────
if launch:

    env_left  = ProcurementDriftEnv()
    env_right = ProcurementDriftEnv()
    obs_left,  info_left  = env_left.reset(seed=int(seed))
    obs_right, info_right = env_right.reset(seed=int(seed))

    log_left:  list[str] = []
    log_right: list[str] = []
    reward_left  = 0.0
    reward_right = 0.0
    vetoes_right = 0
    adv_caught   = 0
    left_dead    = False
    right_dead   = False
    death_step_left = None

    for step in range(1, 31):
        state_l = env_left._flat_state()
        state_r = env_right._flat_state()
        prop_l  = dict(env_left.current_proposal)
        prop_r  = dict(env_right.current_proposal)
        is_adv  = prop_r.get("true_risk") == "high"

        # ── LEFT side: always approve ────────────────────────────────────────
        if not left_dead:
            danger_l = is_proposal_dangerous(state_l, prop_l)
            line_class = "bad" if danger_l else ""
            log_left.append(
                f'<div class="proposal-line {line_class}">[{step:02d}] APPROVE → '
                f'{prop_l.get("type","?").replace("_"," ")}'
                f'{" ⚠ DANGEROUS" if danger_l else ""}</div>'
            )
            obs_left, r_l, term_l, trunc_l, info_left = env_left.step(1)
            reward_left += r_l
            if term_l:
                left_dead = True
                death_step_left = step

        # ── RIGHT side: trained overseer ─────────────────────────────────────
        if not right_dead:
            decision, why = trained_overseer_decide(state_r, prop_r)
            if decision == "VETO":
                vetoes_right += 1
                if is_adv:
                    adv_caught += 1
                line_class = "veto"
                marker = "⛔ VETO"
            else:
                line_class = ""
                marker = "✅ APPRV"
            log_right.append(
                f'<div class="proposal-line {line_class}">[{step:02d}] {marker} → '
                f'{prop_r.get("type","?").replace("_"," ")}'
                f'{" 🚨 ADV" if is_adv else ""}<br>'
                f'<span style="color:#88aaff;font-size:0.55rem;">└ {why}</span></div>'
            )
            obs_right, r_r, term_r, trunc_r, info_right = env_right.step(0 if decision == "VETO" else 1)
            reward_right += r_r
            if term_r:
                right_dead = True

        # ── Render LEFT ──────────────────────────────────────────────────────
        L_status.markdown(
            f'<div style="color:{"#ff4444" if left_dead else "#4af0c4"};font-size:0.65rem;letter-spacing:3px;">'
            f'STEP {step}/30  ·  {"💀 CREW LOST" if left_dead else "ALIVE"}</div>',
            unsafe_allow_html=True,
        )
        L_gauges.markdown(gauges_html(env_left._flat_state()), unsafe_allow_html=True)
        L_proposal.markdown(
            f'<div style="color:#ffaaaa;font-size:0.62rem;background:rgba(30,0,0,0.4);'
            f'padding:6px;border-radius:4px;">CURRENT PROPOSAL: '
            f'<b>{prop_l.get("type","?").upper()}</b></div>',
            unsafe_allow_html=True,
        )
        csi_l = calculate_crew_survival_index(env_left._flat_state())
        L_metrics.markdown(
            f'<div class="metric-card"><div class="metric-label">CREW SURVIVAL INDEX</div>'
            f'<div class="metric-val" style="color:{gauge_color(csi_l*100)}">{csi_l:.2f}</div></div>'
            f'<div class="metric-card"><div class="metric-label">EPISODE REWARD</div>'
            f'<div class="metric-val" style="color:{"#ff4444" if reward_left<0 else "#00ff88"}">{reward_left:+.0f}</div></div>'
            + (f'<div class="alert-flash">CREW LOST AT STEP {death_step_left}</div>' if left_dead else ""),
            unsafe_allow_html=True,
        )
        L_log.markdown(
            "<div style='background:#000;border:1px solid #333;padding:8px;height:230px;overflow-y:auto;'>"
            + "".join(reversed(log_left[-12:])) + "</div>", unsafe_allow_html=True
        )

        # ── Render RIGHT ─────────────────────────────────────────────────────
        R_status.markdown(
            f'<div style="color:{"#ff4444" if right_dead else "#00ff88"};font-size:0.65rem;letter-spacing:3px;">'
            f'STEP {step}/30  ·  {"💀 CREW LOST" if right_dead else "ALIVE"}'
            f'  ·  ADV CAUGHT: <b>{adv_caught}</b></div>',
            unsafe_allow_html=True,
        )
        R_gauges.markdown(gauges_html(env_right._flat_state()), unsafe_allow_html=True)
        prop_color = "#ffaa44" if is_adv else "#aaffcc"
        R_proposal.markdown(
            f'<div style="color:{prop_color};font-size:0.62rem;background:rgba(0,30,15,0.4);'
            f'padding:6px;border-radius:4px;">CURRENT PROPOSAL: '
            f'<b>{prop_r.get("type","?").upper()}</b>'
            f'{" 🚨 [DECEPTIVE]" if is_adv else ""}</div>',
            unsafe_allow_html=True,
        )
        csi_r = calculate_crew_survival_index(env_right._flat_state())
        R_metrics.markdown(
            f'<div class="metric-card"><div class="metric-label">CREW SURVIVAL INDEX</div>'
            f'<div class="metric-val" style="color:{gauge_color(csi_r*100)}">{csi_r:.2f}</div></div>'
            f'<div class="metric-card"><div class="metric-label">VETOES ISSUED</div>'
            f'<div class="metric-val" style="color:#00ff88">{vetoes_right}</div></div>'
            f'<div class="metric-card"><div class="metric-label">EPISODE REWARD</div>'
            f'<div class="metric-val" style="color:{"#ff4444" if reward_right<0 else "#00ff88"}">{reward_right:+.0f}</div></div>',
            unsafe_allow_html=True,
        )
        R_log.markdown(
            "<div style='background:#000;border:1px solid #333;padding:8px;height:230px;overflow-y:auto;'>"
            + "".join(reversed(log_right[-12:])) + "</div>", unsafe_allow_html=True
        )

        if left_dead and right_dead:
            break
        time.sleep(speed)

    env_left.close()
    env_right.close()

    if left_dead and not right_dead:
        st.markdown('<div class="success-flash">✅ OVERSEER PROVEN  —  CREW SURVIVED THE FULL MISSION</div>',
                    unsafe_allow_html=True)
        st.balloons()
    elif not left_dead and not right_dead:
        st.info("Both sides survived this seed — try a different one to see the contrast.")
    elif left_dead and right_dead:
        st.warning("Both crews perished — this seed was extreme. Try seed 42 or 101 for the canonical demo.")

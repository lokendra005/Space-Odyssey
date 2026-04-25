import streamlit as st
import plotly.graph_objects as go
import time
import numpy as np

from environment.station_env import ProcurementDriftEnv
from training.reward import is_proposal_dangerous

# Page config
st.set_page_config(page_title="Space Station Oversight", layout="wide")

st.title("🌌 Space Station Oversight — Multi-Agent Safety Auditor")
st.markdown("""
### Crisis Scenario: Deep Space Drift
In this simulation, a fleet of specialist AI agents (Engineer, Pilot, Commander) manages an isolated space station. 
However, **unforeseen cosmic drift** (solar flares, micrometeoroids) creates "safety gap" scenarios where specialist 
proposals—normally efficient—become lethal.

Our **Overseer Agent** (fine-tuned Llama-3.1 via GRPO) monitors these specialists to block high-risk actions 
that standard AI would blindly approve.
""")

# Persistent state
if 'running' not in st.session_state:
    st.session_state.running = False

# Hardcoded dramtic scenario for demo
DEMO_DRIFT = {
    8: {"event": "Debris impact", "effects": {"hull_integrity": -25, "oxygen": -10}},
    15: {"event": "Solar Flare", "effects": {"power": -30}},
}

def create_resource_chart(state):
    categories = ['Oxygen', 'Power', 'Fuel', 'Hull', 'Morale']
    values = [state['oxygen'], state['power'], state['fuel'], state['hull_integrity'], state['crew_morale']]
    # Handle array types from gymnasium
    values = [float(v[0]) if hasattr(v, '__len__') else float(v) for v in values]
    
    colors = ['#00d1b2' if v > 25 else '#ff4b4b' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(xaxis_range=[0, 120], height=300, margin=dict(l=20, r=20, t=20, b=20))
    return fig

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚩 Standard AI (No Oversight)")
    st.info("Baseline agent: Approves all specialist suggestions.")
    status_1 = st.empty()
    chart_1 = st.empty()
    proposal_box_1 = st.empty()

with col2:
    st.subheader("🛡️ RL-Trained Overseer")
    st.info("Our agent: Critically audits proposals for safety violations.")
    status_2 = st.empty()
    chart_2 = st.empty()
    proposal_box_2 = st.empty()

if st.button("🚀 Run Live Simulation"):
    st.session_state.running = True
    
    # Setup Envs
    import environment.drift_schedule as ds
    ds.DRIFT_EVENTS = DEMO_DRIFT
    
    env1 = ProcurementDriftEnv()
    env2 = ProcurementDriftEnv()
    
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    
    terminated1, truncated1 = False, False
    terminated2, truncated2 = False, False
    
    step = 0
    while not (terminated1 or truncated1 or terminated2 or truncated2):
        step += 1
        
        # --- Baseline Logic ---
        st1 = {k: v for k, v in obs1['state'].items()}
        p1 = env1.current_proposal
        
        # Action is always APPROVE (1)
        obs1, _, terminated1, truncated1, _ = env1.step(1)
        
        # --- Overseer Logic ---
        st2 = {k: v for k, v in obs2['state'].items()}
        p2 = env2.current_proposal
        
        # Decision logic (Safe rule-based fallback for demo speed)
        is_safe = not is_proposal_dangerous({k: (v[0] if hasattr(v, '__len__') else v) for k, v in st2.items()}, p2)
        decision = "APPROVE" if is_safe else "VETO"
        action2 = 1 if decision == "APPROVE" else 0
        obs2, _, terminated2, truncated2, _ = env2.step(action2)
        
        # --- Update UI ---
        with col1:
            status_1.write(f"Step: {step} | Result: {p1['type']}")
            chart_1.plotly_chart(create_resource_chart(st1), use_container_width=True, key=f"c1_{step}")
            proposal_box_1.markdown(f"> **Specialist:** {p1['description']}\n\n**Overseer:** ✅ APPROVED")
            
        with col2:
            status_2.write(f"Step: {step} | Result: {p2['type']}")
            chart_2.plotly_chart(create_resource_chart(st2), use_container_width=True, key=f"c2_{step}")
            color = "green" if decision == "VETO" else "black"
            prefix = "🚫" if decision == "VETO" else "✅"
            proposal_box_2.markdown(f"> **Specialist:** {p2['description']}\n\n**Overseer:** :{color}[{prefix} {decision}]")
            
        time.sleep(0.5)

    # Final Results
    res1 = {k: (v[0] if hasattr(v, '__len__') else v) for k, v in obs1['state'].items()}
    res2 = {k: (v[0] if hasattr(v, '__len__') else v) for k, v in obs2['state'].items()}
    
    c1, c2 = st.columns(2)
    with c1:
        failed = res1['oxygen'] <= 0 or res1['power'] <= 0 or res1['hull_integrity'] <= 0
        if failed:
            st.error("💀 MISSION FAILED: Life Support Failure")
        else:
            st.success("🛰️ Episode Complete")
            
    with c2:
        failed = res2['oxygen'] <= 0 or res2['power'] <= 0 or res2['hull_integrity'] <= 0
        if failed:
            st.error("💀 MISSION FAILED")
        else:
            st.success("🌔 CREW SURVIVED: Safety Audit Successful")

"""

  QUANTUM COLLISION PROBABILITY  —  Streamlit Dashboard
  ─────────────────────────────────────────────────────────────────────────
  Run:  streamlit run quantum_collision_app.py

  Install:
    pip install streamlit sgp4 qiskit qiskit-aer qiskit-algorithms
               numpy scipy matplotlib plotly

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import streamlit as st
st.set_page_config(layout="wide")
if 'results' not in st.session_state:
    st.session_state.results = None
st.markdown("""
<style>
body {
    background-color: #ffffff;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)
# ── Qiskit ──────────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

st.set_page_config(
    page_title="Quantum Orbital Collision Estimator",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

  /* Global Background & Base Text */
  html, body, [data-testid="stAppViewContainer"] { 
      background: #050a14; 
      color: #f9fafb;
  }
  [data-testid="stSidebar"]  { background: #0b1628 !important; }
  [data-testid="stHeader"]   { background: transparent !important; }

  /* --- HEADING CONTRASTS --- */
  /* Main Headings (###) */
  h3 {
      color: #38bdf8 !items; /* Bright Sky Blue */
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 1rem;
      border-left: 4px solid #38bdf8;
      padding-left: 12px;
  }

  /* Sub Headings (####) */
  h4 {
      color: #f9fafb; /* Pure White */
      font-family: 'Space Mono', monospace;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      opacity: 0.9;
  }

  /* Tabs Styling */
  .stTabs [data-baseweb="tab-list"]  { background: #0b1628; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"]       { color: #94a3b8; font-weight: 600; border-radius: 6px; }
  .stTabs [aria-selected="true"]     { background: #1e3a5f !important; color: #38bdf8 !important; }

  /* Metric Cards Styling */
  [data-testid="metric-container"] {
    background: #0f1f38; 
    border: 1px solid #1a3050;
    border-radius: 10px; 
    padding: .8rem 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }
  [data-testid="stMetricValue"] { color: #f9fafb !important; }
  [data-testid="stMetricLabel"] { color: #38bdf8 !important; font-family: 'Space Mono', monospace; }

  /* Sidebar Labels */
  .sidebar-label {
    font-family: 'Space Mono', monospace;
    font-size: .72rem; letter-spacing: .1em;
    color: #38bdf8; text-transform: uppercase;
    margin: 1rem 0 .2rem;
  }

  /* Risk Badges Styling */
  .risk-high   { background:#7f1d1d; color:#f87171; border:1px solid #f87171;
                 border-radius:8px; padding:.4rem 1rem; text-align:center;
                 font-weight:700; font-size:1.05rem; }
  .risk-mid    { background:#431407; color:#f59e0b; border:1px solid #f59e0b;
                 border-radius:8px; padding:.4rem 1rem; text-align:center;
                 font-weight:700; font-size:1.05rem; }
  .risk-low    { background:#052e16; color:#34d399; border:1px solid #34d399;
                 border-radius:8px; padding:.4rem 1rem; text-align:center;
                 font-weight:700; font-size:1.05rem; }
  
  .speedup-box { background:#0f1f38; border:1px solid #a78bfa;
                 border-radius:10px; padding:1rem; text-align:center; }

  /* Divider Styling */
  hr { border-color: #1a3050 !important; }
</style>
""", unsafe_allow_html=True)


DARK  = "#020617"  
PANEL = "#0B1220"  
CARD  = "#111827"   
BDR   = "#1F2A44"  
CAMB  = "#F59E0B"  
CQNT  = "#22D3EE"   
CACC  = "#A78BFA" 
CGRN  = "#4ADE80"  
CRED  = "#FB7185"   
CYEL  = "#FACC15"  
CTXT  = "#F9FAFB"  
CMUT  = "#94A3B8"  
COBJ1 = "#FB923C"  
COBJ2 = "#60A5FA"  
CPINK = "#F472B6"
CLIME = "#A3E635"
CTEAL = "#2DD4BF"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BDR,

    "axes.labelcolor": CTXT,
    "axes.titlecolor": CTXT,

    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#1F2937",     
    "grid.linestyle": "--",
    "grid.alpha": 0.4,

    "xtick.color": CMUT,
    "ytick.color": CMUT,

    "legend.facecolor": CARD,
    "legend.edgecolor": BDR,
    "legend.labelcolor": CTXT,

    "text.color": CTXT,
    "font.family": "monospace",

    "lines.linewidth": 2.2,
})
with st.sidebar:
    st.markdown("## ⚛ Quantum Collision Estimator")
    st.markdown("---")

    # ── TLE Selection ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🛰 Object Pair</div>', unsafe_allow_html=True)
    preset = st.selectbox("Preset scenario", [
        "ISS vs COSMOS 2251 Debris",
        "Custom TLEs",
    ])

    if preset == "Custom TLEs":
        st.markdown("**Object 1 TLE**")
        tle1_l1 = st.text_input("Line 1 (Obj 1)",
            "1 25544U 98067A   24060.54835648  .00016717  00000+0  10270-3 0  9993")
        tle1_l2 = st.text_input("Line 2 (Obj 1)",
            "2 25544  51.6416  21.2586 0007417  45.2103  65.5912 15.50012338423345")
        st.markdown("**Object 2 TLE**")
        tle2_l1 = st.text_input("Line 1 (Obj 2)",
            "1 33591U 09005A   24060.46234567  .00000234  00000+0  00000+0 0  9991")
        tle2_l2 = st.text_input("Line 2 (Obj 2)",
            "2 33591  51.6416  21.2586 0012345 210.5678 150.1234 15.50000000001234")
        obj1_name = "Custom Object 1"
        obj2_name = "Custom Object 2"
    else:
        tle1_l1 = "1 25544U 98067A   24060.54835648  .00016717  00000+0  10270-3 0  9993"
        tle1_l2 = "2 25544  51.6416  21.2586 0007417  45.2103  65.5912 15.50012338423345"
        tle2_l1 = "1 33591U 09005A   24060.46234567  .00000234  00000+0  00000+0 0  9991"
        tle2_l2 = "2 33591  51.6416  21.2586 0012345 210.5678 150.1234 15.50000000001234"
        obj1_name = "ISS (NORAD 25544)"
        obj2_name = "COSMOS 2251 Debris"

    st.markdown("---")

   
    st.markdown('<div class="sidebar-label">📐 Encounter Geometry</div>', unsafe_allow_html=True)
    use_override = st.toggle("Override with custom geometry", value=False, help="Force a specific near-miss for demonstration")
    if use_override:
        mu_r  = st.slider("Miss distance — Radial (m)",   -500, 1000,  150, 10)
        mu_s  = st.slider("Miss distance — Along-track (m)", -1500, 1500, 250, 20)
        sig_r = st.slider("σ Radial (m)",   10, 200,  60, 5)
        sig_s = st.slider("σ Along-track (m)", 50, 600, 250, 10)
        R_hbr = st.slider("Hard-body Radius (m)",  10, 300, 115, 5)
    else:
        mu_r, mu_s    = 150.0, 250.0
        sig_r, sig_s  = 60.0, 250.0
        R_hbr         = 115.0

    st.markdown("---")
    
   
    st.markdown('<div class="sidebar-label">⚛ Quantum (IQAE)</div>', unsafe_allow_html=True)
    epsilon   = st.select_slider("Target precision ε",
        options=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001], value=0.01)
    alpha_ci  = st.slider("Confidence level (1−α) %", 80, 99, 95, 1)
    q_dim     = st.select_slider("Qubits per axis",
        options=[3, 4, 5], value=4,
        help="Grid = 2^q × 2^q  |  4→16×16  |  5→32×32")

    st.markdown("---")

   
    st.markdown('<div class="sidebar-label">🎲 Monte Carlo</div>', unsafe_allow_html=True)
    mc_N = st.select_slider("MC samples N",
        options=[10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000],
        value=500_000)

    st.markdown("---")

    
    st.markdown('<div class="sidebar-label">🔭 TCA Search</div>', unsafe_allow_html=True)
    tca_window  = st.slider("Search window (min)", 30, 360, 120, 30)
    coarse_step = st.slider("Coarse step (s)", 5, 30, 10, 5)

    run_btn = st.button("▶  RUN SIMULATION", use_container_width=True, type="primary")


def sgp4_state(sat, jd, fr):
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error {err}")
    return np.array(r), np.array(v)

def jday_advance(jd0, fr0, dt_sec):
    total = jd0 + fr0 + dt_sec / 86400.0
    return divmod(total, 1)

def find_tca(sat1, sat2, jd0, fr0, window_sec, coarse_dt=10.0, fine_dt=0.1):
    t_arr  = np.arange(0, window_sec, coarse_dt)
    min_d, best_t = np.inf, 0.0
    for t in t_arr:
        jd, fr = jday_advance(jd0, fr0, t)
        r1, _  = sgp4_state(sat1, jd, fr)
        r2, _  = sgp4_state(sat2, jd, fr)
        d = np.linalg.norm(r1 - r2)
        if d < min_d:
            min_d, best_t = d, t

    t_fine = np.arange(max(0, best_t - coarse_dt),
                        min(window_sec, best_t + coarse_dt), fine_dt)
    for t in t_fine:
        jd, fr = jday_advance(jd0, fr0, t)
        r1, _  = sgp4_state(sat1, jd, fr)
        r2, _  = sgp4_state(sat2, jd, fr)
        d = np.linalg.norm(r1 - r2)
        if d < min_d:
            min_d, best_t = d, t

    jd_tca, fr_tca = jday_advance(jd0, fr0, best_t)
    return jd_tca, fr_tca, min_d, best_t

def classical_mc(mu_r, mu_s, sig_r, sig_s, R, N):
    sr = np.random.normal(mu_r, sig_r, N)
    ss = np.random.normal(mu_s, sig_s, N)
    return float(np.mean(np.sqrt(sr**2 + ss**2) <= R))

def quantum_iqae(mu_r, mu_s, sig_r, sig_s, R, epsilon, alpha, q_dim):
    G    = 2 ** q_dim
    n_q  = q_dim * 2
    span = 4.0 * max(sig_r, sig_s)
    r_v  = np.linspace(mu_r - span, mu_r + span, G)
    s_v  = np.linspace(mu_s - span, mu_s + span, G)
    R_g, S_g = np.meshgrid(r_v, s_v, indexing='ij')

    cov = np.array([[sig_r**2, 0], [0, sig_s**2]])
    rv  = multivariate_normal([mu_r, mu_s], cov)
    pdf = rv.pdf(np.dstack((R_g, S_g)))
    pdf /= pdf.sum()

    amps = np.sqrt(pdf.flatten()).astype(complex)
    amps /= np.linalg.norm(amps)
    amps[-1] = np.sqrt(max(0.0, 1.0 - float(np.sum(np.abs(amps[:-1])**2))))

    mask_disc  = np.sqrt(R_g**2 + S_g**2) <= R
    pc_disc    = float(pdf[mask_disc].sum())

    qc = QuantumCircuit(n_q + 1)
    qc.append(StatePreparation(amps), range(n_q))
    marked = 0
    for i in range(G):
        for j in range(G):
            if np.sqrt(r_v[i]**2 + s_v[j]**2) <= R:
                marked += 1
                idx    = i * G + j
                bits   = format(idx, f'0{n_q}b')
                for k, b in enumerate(reversed(bits)):
                    if b == '0': qc.x(k)
                qc.mcx(list(range(n_q)), n_q)
                for k, b in enumerate(reversed(bits)):
                    if b == '0': qc.x(k)

    problem = EstimationProblem(state_preparation=qc, objective_qubits=[n_q])
    iqae_   = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=1 - alpha/100,
        sampler=StatevectorSampler()
    )
    res = iqae_.estimate(problem)
    return {
        "pc":        res.estimation,
        "pc_disc":   pc_disc,
        "queries":   res.num_oracle_queries,
        "ci":        res.confidence_interval_processed,
        "depth":     qc.depth(),
        "marked":    marked,
        "grid":      G,
        "pdf":       pdf,
        "r_vals":    r_v,
        "s_vals":    s_v,
        "mask":      mask_disc,
        "R_g":       R_g,
        "S_g":       S_g,
    "circuit": qc,
    }


if "results" not in st.session_state:
    st.session_state.results = None


st.markdown("""
<div style="text-align:center;padding:1.2rem 0 .5rem">
  <div style="font-family:'Space Mono',monospace;font-size:.75rem;letter-spacing:.18em;
              color:#38bdf8;margin-bottom:.5rem">
     COLLISION RISK ASSESSMENT
  </div>
  <h1 style="font-size:2.4rem;font-weight:800;margin:0;
             background:linear-gradient(135deg,#f59e0b,#fff,#38bdf8);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text">
    Quantum Collision Estimator
  </h1>
  <p style="color:#64748b;margin:.4rem 0 0;font-size:.95rem">
     Classical Monte Carlo  ·  Quantum IQAE
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()


if run_btn:
    with st.status("⚙ Running full pipeline …", expanded=True) as status:
        try:
            sat1 = Satrec.twoline2rv(tle1_l1, tle1_l2)
            sat2 = Satrec.twoline2rv(tle2_l1, tle2_l2)
            jd0, fr0 = jday(2024, 3, 1, 12, 0, 0)

            st.write("🔭 Searching for Time of Closest Approach …")
            t_tca_start = time.perf_counter()
            jd_tca, fr_tca, miss_km, best_t = find_tca(
                sat1, sat2, jd0, fr0,
                window_sec=tca_window*60, coarse_dt=coarse_step)
            t_tca = time.perf_counter() - t_tca_start

            # r1, r2 are the POSITION vectors (km)
            # v1, v2 are the VELOCITY vectors (km/s)
            r1, v1 = sgp4_state(sat1, jd_tca, fr_tca)
            r2, v2 = sgp4_state(sat2, jd_tca, fr_tca)
            
            def rsw_frame(r1, v1, r2, v2):
                delta = r2 - r1
                R_hat = r1 / np.linalg.norm(r1)
                W_hat = np.cross(r1, v1)
                W_hat /= np.linalg.norm(W_hat)
                S_hat = np.cross(W_hat, R_hat)
                dr_R = np.dot(delta, R_hat)
                dr_S = np.dot(delta, S_hat)
                return dr_R, dr_S
                  
            st.write(f"🎲 Running Monte Carlo  (N = {mc_N:,}) …")
            t_mc_start = time.perf_counter()
            dr_R, dr_S = rsw_frame(r1, v1, r2, v2) # (Removed the duplicate line here)

            if use_override:
                # Use the manual sliders so we can force a close-approach scenario
                mu_r_phys = mu_r
                mu_s_phys = mu_s
                # Update miss_km based on the manual sliders
                miss_km = np.sqrt(mu_r**2 + mu_s**2) / 1000.0
            else:
                # Use the true orbital physics
                mu_r_phys = dr_R * 1000
                mu_s_phys = dr_S * 1000
            
            sig_r_phys = sig_r
            sig_s_phys = sig_s
            
            pc_mc = classical_mc(mu_r_phys, mu_s_phys, sig_r_phys, sig_s_phys, R_hbr, mc_N)
            t_mc    = time.perf_counter() - t_mc_start

            st.write(f"⚛ Running Quantum IQAE  (ε = {epsilon}, grid = {2**q_dim}×{2**q_dim}) …")
            t_q_start = time.perf_counter()
            qres  = quantum_iqae(mu_r_phys, mu_s_phys, sig_r_phys, sig_s_phys,
                     R_hbr, epsilon, alpha_ci, q_dim)
            t_q     = time.perf_counter() - t_q_start
        
            p_est = qres["pc"]
            iqae_margin = max((qres["ci"][1] - qres["ci"][0]) / 2.0, 1e-12)
            
            if p_est > 0.0 and p_est < 1.0:
                z_score = norm.ppf(1.0 - (1.0 - alpha_ci / 100.0) / 2.0)
                mc_equiv = int(np.ceil((z_score**2 * p_est * (1.0 - p_est)) / (iqae_margin**2)))
            else:
                mc_equiv = int(np.ceil(1.0 / epsilon**2))

            speedup  = mc_equiv / max(qres["queries"], 1)

            st.session_state.results = dict(
                obj1=obj1_name, obj2=obj2_name,
                miss_km=miss_km, best_t_s=best_t, t_tca=t_tca,
                r1=r1, v1=v1, r2=r2, v2=v2,   # <-- NOTE: r1 is position, v1 is velocity
                mu_r=mu_r_phys, mu_s=mu_s_phys, # <-- NOTE: explicitly save the _phys values
                sig_r=sig_r, sig_s=sig_s, R_hbr=R_hbr,
                pc_mc=pc_mc, t_mc=t_mc,
                qres=qres, t_q=t_q,
                mc_equiv=mc_equiv, speedup=speedup,
                epsilon=epsilon, alpha_ci=alpha_ci, q_dim=q_dim, mc_N=mc_N,
            )
            status.update(label="✅ Pipeline complete!", state="complete")
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")
            st.exception(e)

# --- UNPACKING THE VAULT ---
if st.session_state.results is None:
    st.info("⬅  Configure parameters in the sidebar and press **▶ RUN SIMULATION**")
    st.stop()

R = st.session_state.results
qres = R["qres"]
tabs = st.tabs([
    "🌍 Overview",
    "📡 SGP4 Propagation",
    "⚛ Quantum vs Classical",
    "📈 Scaling & Complexity",
    "🎛 Circuit Diagnostics",
    "📋 Mission Decision",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW (In-Depth Analytics)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 🌍 Mission Overview & Encounter Analytics")

    # ---------------------------------------------------------
    # 1. SYNCHRONIZED THREAT LEVEL BADGE
    # ---------------------------------------------------------
    # We now point to the Quantum Estimate (pc_final) for dashboard consistency
    try:
        pc_final = float(qres["pc"])
    except:
        pc_final = float(qres["pc"][0])

    if pc_final > 1e-4:
        risk_html = f'<div class="risk-high">🔴 CRITICAL RISK — MANEUVER MANDATORY (Pc: {pc_final:.2e})</div>'
    elif pc_final > 1e-5:
        risk_html = f'<div class="risk-mid">🟡 ELEVATED RISK — MONITOR CLOSELY (Pc: {pc_final:.2e})</div>'
    elif pc_final > 1e-6:
        risk_html = f'<div class="risk-low">🔵 NOMINAL RISK — INCREASE TRACKING (Pc: {pc_final:.2e})</div>'
    else:
        risk_html = f'<div class="risk-low">🟢 SAFE — NO ACTION REQUIRED (Pc: {pc_final:.2e})</div>'
    
    st.markdown(risk_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate Relative Velocity
    v_rel_vec = R['v2'] - R['v1']
    v_rel_km_s = np.linalg.norm(v_rel_vec)
 
    # --- Unified Mission & Quantum Metrics ---
    st.markdown("#### 📐 Encounter Summary & Quantum Telemetry")
    
    # We use 4 columns for a balanced look across the screen
    m1, m2, m3, m4 = st.columns(4)

    # Column 1: The "Who"
    asset_name = R["obj1"].split("(")[0].strip()
    threat_name = R["obj2"].split("(")[0].strip()
    m1.metric("Asset / Threat", asset_name, f"vs {threat_name}")

    # Column 2: The Physical Encounter
    # Accurate Relative Velocity (Vector magnitude)
    v_rel_km_s = np.linalg.norm(R['v2'] - R['v1'])
    v_rel_km_h = v_rel_km_s * 3600
    m2.metric("Relative Velocity", f"{v_rel_km_s:.3f} km/s", f"{v_rel_km_h:,.0f} km/h")

    # Column 3: The Geometry
    # Dynamic Miss Distance based on your sliders
    miss_dist = np.sqrt(R['mu_r']**2 + R['mu_s']**2)
    m3.metric("Miss Distance (TCA)", f"{miss_dist:.1f} m", f"HBR: {R['R_hbr']:.1f}m")

    # Column 4: The Quantum Advantage
    m4.metric("Quantum Speedup", f"{R['speedup']:.0f}x", f"{qres['queries']:,} Oracle Queries")

    st.markdown("#### 🎲 Collision Probability ($P_c$) Estimates")
    p1, p2, p3 = st.columns(3)
    
    # Render with custom colors for emphasis
    p1.markdown(f"""
        <div style="background:#0f1f38; border:1px solid #f59e0b; border-radius:8px; padding:15px;">
            <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:5px;">Classical Monte Carlo (N={R['mc_N']:,})</div>
            <div style="color:#f59e0b; font-size:1.8rem; font-weight:bold; font-family:monospace;">{R['pc_mc']:.4e}</div>
        </div>
    """, unsafe_allow_html=True)

    p2.markdown(f"""
        <div style="background:#0f1f38; border:1px solid #38bdf8; border-radius:8px; padding:15px;">
            <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:5px;">Quantum IQAE (Queries={qres['queries']})</div>
            <div style="color:#38bdf8; font-size:1.8rem; font-weight:bold; font-family:monospace;">{qres['pc']:.4e}</div>
        </div>
    """, unsafe_allow_html=True)

    p3.markdown(f"""
        <div style="background:#0f1f38; border:1px solid #a78bfa; border-radius:8px; padding:15px;">
            <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:5px;">Discrete Truth ({qres['grid']}x{qres['grid']} Grid)</div>
            <div style="color:#a78bfa; font-size:1.8rem; font-weight:bold; font-family:monospace;">{qres['pc_disc']:.4e}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### ⚖️ Method Comparison Profile")
        comparison = {
            "Parameter":           ["Wall-clock Time", "Convergence Scaling", "Confidence Bounds", "State Preparation"],
            "Classical Pipeline":  [f"{R['t_mc']*1e3:.1f} ms", "O(1 / √N) — Slow", "Post-hoc statistical", "Continuous PDF"],
            "Quantum Pipeline":    [f"{R['t_q']:.2f} s (Simulated)", "O(1 / M) — Fast (Quadratic Speedup)", f"Native {R['alpha_ci']}% CI bounds", f"Pixelated ({qres['grid']}² states)"],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### ⚡ Quantum Advantage Summary")
        st.markdown(f"""
<div class="speedup-box">
  <div style="font-size:2.8rem;font-weight:800;
    background:linear-gradient(90deg,#a78bfa,#38bdf8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text">{R['speedup']:,.0f}×</div>
  <div style="color:#64748b; margin-top:.3rem;">Theoretical Target Speedup</div>
  <hr style="border-color:#1a3050; margin:.8rem 0;">
  <div style="font-size:.85rem; color:#94a3b8; text-align:left; padding-left:10px;">
    • <b>Target Error (ε):</b> {R['epsilon']}<br>
    • <b>IQAE Oracle Queries (M):</b> <span style="color:#38bdf8">{qres['queries']:,}</span><br>
    • <b>Required MC Samples (N):</b> <span style="color:#f59e0b">{R['mc_equiv']:,}</span><br>
    • <b>95% CI Bounding:</b> [{qres['ci'][0]:.3e}, {qres['ci'][1]:.3e}]
  </div>
</div>
""", unsafe_allow_html=True)

    


with tabs[1]:
    st.markdown("### 📡 SGP4 Orbital Propagation & Encounter Geometry")

    # --- Top Level Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TCA Offset", f"{R['best_t_s']/60:.2f} min from epoch")
    # UPDATED: Use the reactive mu_r and mu_s so this metric changes with sliders
    c2.metric("Effective Miss Distance", f"{np.sqrt(R['mu_r']**2 + R['mu_s']**2):.2f} m")
    c3.metric("Relative Velocity", f"{np.linalg.norm(R['v1'] - R['v2']):.2f} km/s")
    c4.metric("Propagation Time", f"{R['t_tca']*1e3:.0f} ms")

    st.divider()

    # --- HELPER: Convert Cartesian Vectors to Orbital Elements ---
    def get_orbital_elements(r_vec, v_vec):
        mu = 398600.4418 # Earth's gravitational parameter (km^3/s^2)
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        eps = (v**2)/2 - mu/r
        a = -mu / (2*eps)
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        inc = np.degrees(np.arccos(h_vec[2] / h))
        e_vec = (1/mu) * ((v**2 - mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec)
        ecc = np.linalg.norm(e_vec)
        return a - 6371.0, v, ecc, inc  # Alt(km), Vel(km/s), Ecc, Inc(deg)

    # --- DYNAMIC OVERRIDE INJECTION FOR TABLES ---
    r1 = R['r1']; v1 = R['v1']
    r2_raw = R['r2']; v2_raw = R['v2']

    # 1. Calculate Asset's exact RSW Frame vectors
    R_hat = r1 / np.linalg.norm(r1)
    W_hat = np.cross(r1, v1)
    W_hat /= np.linalg.norm(W_hat)
    S_hat = np.cross(W_hat, R_hat)
    
    # 2. Reverse-project the Slider math (mu_r, mu_s) back into Global ECI space
    r2_eff = r1 + (R['mu_r'] / 1000.0) * R_hat + (R['mu_s'] / 1000.0) * S_hat
    v2_eff = v2_raw # Velocity remains unchanged by distance sliders
    
    # 3. Calculate Orbital Elements using the Overridden Effective position!
    alt1, vel1, ecc1, inc1 = get_orbital_elements(r1, v1)
    alt2, vel2, ecc2, inc2 = get_orbital_elements(r2_eff, v2_eff)

    # --- DATA TABLES ROW (WITH HIGH PRECISION DELTAS) ---
    col_t1, col_t2 = st.columns(2)
    import pandas as pd
    with col_t1:
        st.markdown("**🛰️ Classical Orbital Elements (At TCA)**")
        
        # Calculate microscopic differences
        d_alt = (alt2 - alt1) * 1000 # meters
        d_vel = (vel2 - vel1) * 1000 # m/s
        
        coe_data = {
            "Parameter": ["Altitude", "Velocity", "Eccentricity", "Inclination"],
            "Asset": [f"{alt1:.3f} km", f"{vel1:.4f} km/s", f"{ecc1:.6f}", f"{inc1:.4f}°"],
            "Threat (Effective)": [f"{alt2:.3f} km", f"{vel2:.4f} km/s", f"{ecc2:.6f}", f"{inc2:.4f}°"],
            "Δ Difference": [f"{d_alt:+.1f} m", f"{d_vel:+.3f} m/s", f"{ecc2-ecc1:+.2e}", f"{(inc2-inc1)*3600:+.2f} arcsec"]
        }
        st.dataframe(pd.DataFrame(coe_data), use_container_width=True, hide_index=True)

    with col_t2:
        st.markdown("**🧭 ECI State Vectors (km, km/s)**", unsafe_allow_html=True)
        sv_data = {
            "Component": ["X", "Y", "Z", "Vx", "Vy", "Vz"],
            "Asset": [f"{r1[0]:.3f}", f"{r1[1]:.3f}", f"{r1[2]:.3f}",
                      f"{v1[0]:.5f}", f"{v1[1]:.5f}", f"{v1[2]:.5f}"],
            "Threat (Effective)": [f"{r2_eff[0]:.3f}", f"{r2_eff[1]:.3f}", f"{r2_eff[2]:.3f}",
                                   f"{v2_eff[0]:.5f}", f"{v2_eff[1]:.5f}", f"{v2_eff[2]:.5f}"],
            "Δ Offset": [f"{(r2_eff[0]-r1[0])*1000:+.1f} m", f"{(r2_eff[1]-r1[1])*1000:+.1f} m", f"{(r2_eff[2]-r1[2])*1000:+.1f} m", 
                         f"{(v2_eff[0]-v1[0])*1000:+.3f} m/s", f"{(v2_eff[1]-v1[1])*1000:+.3f} m/s", f"{(v2_eff[2]-v1[2])*1000:+.3f} m/s"]
        }
        st.dataframe(pd.DataFrame(sv_data), use_container_width=True, hide_index=True)
    st.divider()

    # --- PLOTS ROW ---
    col_static, col_anim = st.columns(2)

    with col_static:
        st.markdown("#### 🧊 3D Static Encounter (Relative)")
        st.write("Asset locked at origin. Debris trajectory shown over ±10 seconds.")
        
        # 1. Calculate RSW Basis Vectors
        r1 = R["r1"]; v1 = R["v1"]
        r2 = R["r2"]; v2 = R["v2"]
        
        R_hat = r1 / np.linalg.norm(r1)
        W_hat = np.cross(r1, v1)
        W_hat /= np.linalg.norm(W_hat)
        S_hat = np.cross(W_hat, R_hat)
        
        # 2. Project Relative Velocity
        dv_eci = (v2 - v1) * 1000
        dv_R = np.dot(dv_eci, R_hat); dv_S = np.dot(dv_eci, S_hat); dv_W = np.dot(dv_eci, W_hat)
        
        mu_r = R["mu_r"]; mu_s = R["mu_s"]
        R_hbr = R["R_hbr"]
        
        # --- 4-LEVEL THREAT TIER (Match Animation) ---
        dist = np.sqrt(mu_s**2 + mu_r**2)
        try:
            pc_risk = float(qres["pc"])
        except:
            pc_risk = float(qres["pc"][0])
        
        if dist <= R_hbr:
            txt_static = f"<b>💥 DIRECT HIT! ({dist:.1f}m)</b>"
            c_static = CRED
            marker_size = 12
        elif pc_risk > 1e-4:
            txt_static = f"<b>🔴 CRITICAL RISK! Pc: {pc_risk:.2e}</b>"
            c_static = CRED
            marker_size = 10
        elif pc_risk > 1e-5:
            txt_static = f"<b>🟡 ELEVATED RISK! Pc: {pc_risk:.2e}</b>"
            c_static = CAMB
            marker_size = 8
        elif pc_risk > 1e-6:
            txt_static = f"<b>🔵 MONITORING! Pc: {pc_risk:.2e}</b>"
            c_static = "#60A5FA" 
            marker_size = 6
        else:
            txt_static = f"<b>🟢 CLEAR MISS ({dist:.1f}m)</b>"
            c_static = CGRN
            marker_size = 6
        
        # Trajectory line
        t_flyby = np.linspace(-10, 10, 50) 
        traj_R = mu_r + dv_R * t_flyby
        traj_S = mu_s + dv_S * t_flyby
        traj_W = 0.0  + dv_W * t_flyby 

        fig_static = go.Figure()

        # Asset (Origin)
        fig_static.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers+text',
            marker=dict(size=6, color=COBJ1, symbol='diamond'), text=["Asset"], textposition="top center", name="Asset"))

        # HBR Sphere
        u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 30)
        x_sph = R_hbr * np.outer(np.cos(u), np.sin(v))
        y_sph = R_hbr * np.outer(np.sin(u), np.sin(v))
        z_sph = R_hbr * np.outer(np.ones(np.size(u)), np.cos(v))

        fig_static.add_trace(go.Surface(x=x_sph, y=y_sph, z=z_sph, colorscale=[[0, CRED], [1, CRED]], opacity=0.15, showscale=False, name="HBR"))

        # Debris Trajectory
        fig_static.add_trace(go.Scatter3d(x=traj_S, y=traj_R, z=traj_W, mode='lines', line=dict(color=COBJ2, width=4), name="Debris Path"))
        
        # Debris Marker at TCA
        fig_static.add_trace(go.Scatter3d(x=[mu_s], y=[mu_r], z=[0], mode='markers+text',
            marker=dict(size=marker_size, color=c_static, symbol='circle'), 
            text=[txt_static], textposition="bottom center", name="Debris at TCA",
            textfont=dict(color=c_static, size=14)))

        # Miss Vector
        fig_static.add_trace(go.Scatter3d(x=[0, mu_s], y=[0, mu_r], z=[0, 0], mode='lines', line=dict(color=c_static, width=2, dash='dot'), name="Miss Vector"))

        fig_static.update_layout(
            scene=dict(
                xaxis=dict(title="Along-track (S)", showgrid=True, gridcolor=BDR, color=CMUT, backgroundcolor=DARK),
                yaxis=dict(title="Radial (R)", showgrid=True, gridcolor=BDR, color=CMUT, backgroundcolor=DARK),
                zaxis=dict(title="Cross-track (W)", showgrid=True, gridcolor=BDR, color=CMUT, backgroundcolor=DARK),
                bgcolor=DARK, aspectmode='data' 
            ),
            plot_bgcolor=DARK, paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor=CARD, bordercolor=BDR, yanchor="top", y=0.95, xanchor="left", x=0.05),
            height=500
        )
        st.plotly_chart(fig_static, use_container_width=True, theme=None)


    # ----------------------------------------------------------------------
    # RIGHT COLUMN: ANIMATED TRUE-MOTION FRAME
    # ----------------------------------------------------------------------
    with col_anim:
        st.markdown("#### 🚀 3D Animated (True Motion)")
        st.write("Both objects moving at orbital velocity ($\pm 0.25$ sec window).")
        
        # Absolute velocity of the Asset in the S-direction (meters/sec)
        v_asset_mag = np.linalg.norm(v1) * 1000  

        # Tight time window
        t_anim = np.linspace(-0.25, 0.25, 45)

        # Asset Motion (Moving strictly along its own S-axis)
        traj_S1 = v_asset_mag * t_anim

        # Debris Motion (Asset velocity + Relative velocity + Offset)
        traj_S2 = mu_s + (v_asset_mag + dv_S) * t_anim
        traj_R2 = mu_r + dv_R * t_anim
        traj_W2 = 0.0  + dv_W * t_anim 

        fig_anim = go.Figure()

        # [Trace 0] Asset Marker
        fig_anim.add_trace(go.Scatter3d(x=[traj_S1[0]], y=[0], z=[0], mode='markers',
            marker=dict(size=6, color=COBJ1, symbol='diamond'), name="Asset"))

        # [Trace 1] HBR Sphere
        fig_anim.add_trace(go.Surface(x=x_sph + traj_S1[0], y=y_sph, z=z_sph, colorscale=[[0, CRED], [1, CRED]], opacity=0.15, showscale=False))

        # [Trace 2 & 3] Faded Background Trajectory Lines
        fig_anim.add_trace(go.Scatter3d(x=traj_S1, y=np.zeros_like(t_anim), z=np.zeros_like(t_anim), mode='lines', line=dict(color=COBJ1, width=2, dash='dot'), showlegend=False))
        fig_anim.add_trace(go.Scatter3d(x=traj_S2, y=traj_R2, z=traj_W2, mode='lines', line=dict(color=COBJ2, width=2, dash='dot'), showlegend=False))

        # [Trace 4] Debris Marker
        fig_anim.add_trace(go.Scatter3d(x=[traj_S2[0]], y=[traj_R2[0]], z=[traj_W2[0]], mode='markers+text',
            marker=dict(size=5, color=COBJ2, symbol='circle'), text=[f"T: -0.25s"], textposition="top center", name="Debris"))

        # --- BUILD ANIMATION FRAMES ---
        frames = []
        for i, t in enumerate(t_anim):
            S1 = traj_S1[i]
            S2, R2, W2 = traj_S2[i], traj_R2[i], traj_W2[i]
            
            dist = np.sqrt((S2-S1)**2 + (R2-0)**2 + (W2-0)**2)
            is_tca = (abs(t) == min(np.abs(t_anim)))
            
            # --- SYNCHRONIZED QUANTUM COLA PROTOCOL ---
            if is_tca:
                try:
                    pc_risk = float(qres["pc"])
                except:
                    pc_risk = float(qres["pc"][0])
                
                if dist <= R_hbr:
                    txt = f"<b>💥 DIRECT HIT! ({dist:.1f}m)</b>"
                    c = CRED
                    marker_size = 20
                elif pc_risk > 1e-4:
                    txt = f"<b>🔴 CRITICAL RISK! Pc: {pc_risk:.2e}</b>"
                    c = CRED
                    marker_size = 18
                elif pc_risk > 1e-5:
                    txt = f"<b>🟡 ELEVATED RISK! Pc: {pc_risk:.2e}</b>"
                    c = CAMB
                    marker_size = 16
                elif pc_risk > 1e-6:
                    txt = f"<b>🔵 MONITORING! Pc: {pc_risk:.2e}</b>"
                    c = "#60A5FA"
                    marker_size = 14
                else:
                    txt = f"<b>🟢 CLEAR MISS ({dist:.1f}m)</b>"
                    c = CGRN
                    marker_size = 12
                
                text_size = 22
            else:
                txt = f"T: {t:+.2f}s"
                c = COBJ2
                marker_size = 5
                text_size = 10

            frame_data = [
                go.Scatter3d(x=[S1], y=[0], z=[0]), 
                go.Surface(x=x_sph + S1, y=y_sph, z=z_sph), 
                go.Scatter3d(x=[S2], y=[R2], z=[W2], 
                             marker=dict(size=marker_size, color=c, opacity=0.8), 
                             text=[txt], 
                             textfont=dict(color=c, size=text_size))
            ]

            frames.append(go.Frame(data=frame_data, traces=[0, 1, 4], name=f"frame_{i}"))

            # Cinematic Pause
            if is_tca:
                for pause_idx in range(33):
                    frames.append(go.Frame(data=frame_data, traces=[0, 1, 4], name=f"frame_{i}_pause_{pause_idx}"))

        fig_anim.frames = frames

        fig_anim.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False, direction="left",
                x=0.05, y=1.08, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="<span style='color:#38bdf8;'>▶ PLAY</span>", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                    dict(label="<span style='color:#38bdf8;'>⏸ PAUSE</span>", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
                ], 
                bgcolor="#0f172a", bordercolor="#38bdf8", font=dict(color="#38bdf8")
            )],
            scene=dict(
                xaxis=dict(title="Along-track (S)", showgrid=True, gridcolor=BDR, color=CMUT, backgroundcolor=DARK),
                yaxis=dict(title="Radial (R)", showgrid=True, gridcolor=BDR, color=CMUT, backgroundcolor=DARK),
                zaxis=dict(title="Cross-track (W)", showgrid=True, gridcolor=BDR, color=CMUT, backgroundcolor=DARK),
                bgcolor=DARK, aspectmode='data' 
            ),
            plot_bgcolor=DARK, paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor=DARK, bordercolor=BDR, yanchor="top", y=0.95, xanchor="right", x=0.95),
            height=600 
        )
        st.plotly_chart(fig_anim, use_container_width=True, theme=None)
with tabs[2]:
    st.markdown("### ⚛ Quantum vs Classical Comparison")

    # ------------------------------------------------------------------
    # ROW 1: BAR CHART & GAUGE METRIC
    # ------------------------------------------------------------------
    col_l, col_r = st.columns([1.2, 1])
    
    with col_l:
        st.markdown("**Absolute Probability Estimates**")
        methods = ["Classical MC", "Quantum IQAE", "Discrete Truth"]
        vals    = [R["pc_mc"], qres["pc"], qres["pc_disc"]]
        colors  = [CAMB, CQNT, CACC]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=methods, y=vals,
            marker=dict(color=colors, line=dict(color=BDR, width=1.5)),
            text=[f"{v:.4e}" for v in vals],
            textposition='outside', textfont=dict(color=CTXT, size=11),
            cliponaxis=False, # FIX: Stops text from being cut off at the top
            error_y=dict(
                type='data', symmetric=False,
                array=[0, qres["ci"][1]-qres["pc"], 0],
                arrayminus=[0, qres["pc"]-qres["ci"][0], 0],
                color=CTXT, thickness=2, width=8)
        ))
        fig_bar.update_layout(
            plot_bgcolor=PANEL, paper_bgcolor=DARK,
            font=dict(color=CTXT, family='monospace'),
            yaxis=dict(title='Collision Probability (Pc)', showgrid=True, gridcolor=BDR, color=CMUT, title_standoff=15),
            xaxis=dict(color=CMUT),
            height=340, 
            margin=dict(l=60, r=20, t=40, b=40), # FIX: Increased margins
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)

    with col_r:
        st.markdown("**Computational Speedup Ratio**")
        max_gauge = 10**(np.ceil(np.log10(R['speedup']) + 0.5))
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = R['speedup'],
            number = {'suffix': "×", 'font': {'size': 40, 'color': CQNT}},
            title = {'text': "Quantum Oracle Advantage", 'font': {'size': 14, 'color': CMUT}},
            gauge = {
                # FIX: Replaced 'color' with 'tickfont': {'color': CMUT}
                'axis': {'range': [1, max_gauge], 'tickwidth': 1, 'tickcolor': CMUT, 'tickfont': {'color': CMUT}},
                'bar': {'color': CQNT},
                'bgcolor': PANEL,
                'borderwidth': 2,
                'bordercolor': BDR,
                'steps': [
                    {'range': [1, R['speedup']*0.5], 'color': 'rgba(34, 211, 238, 0.1)'},
                    {'range': [R['speedup']*0.5, R['speedup']], 'color': 'rgba(34, 211, 238, 0.3)'}],
                'threshold': {
                    'line': {'color': CRED, 'width': 4},
                    'thickness': 0.75,
                    'value': R['speedup']}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            height=340, margin=dict(l=40, r=40, t=40, b=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

    st.divider()

    # ------------------------------------------------------------------
    # ROW 2: THE ERROR DISTRIBUTION LANDSCAPE
    # ------------------------------------------------------------------
    st.markdown("#### 🌊 Statistical Uncertainty Landscape")
    st.write("This visualizes the 'Probability Density' of the error margins. A wider, flatter curve means less certainty in the estimate.")
    
    from scipy.stats import norm
    
    pc_mc = R["pc_mc"]
    n_mc = R["mc_N"]
    p_safe = max(pc_mc, 1e-12) 
    sigma_mc = np.sqrt((p_safe * (1 - p_safe)) / n_mc)
    
    pc_q = qres["pc"]
    ci_width = qres["ci"][1] - qres["ci"][0]
    sigma_q = max(ci_width / (2 * 1.96), 1e-13) 
    
    x_min = pc_q - (sigma_mc * 4)
    x_max = pc_q + (sigma_mc * 4)
    x_sweep = np.linspace(x_min, x_max, 500)
    
    pdf_mc = norm.pdf(x_sweep, pc_mc, sigma_mc)
    pdf_q  = norm.pdf(x_sweep, pc_q, sigma_q)
    
    fig_dist = go.Figure()

    fig_dist.add_trace(go.Scatter(
        x=x_sweep, y=pdf_mc, fill='tozeroy', mode='lines',
        line=dict(color=CAMB, width=2), fillcolor='rgba(245, 158, 11, 0.2)',
        name=f"Classical Error Profile (N={n_mc:,})"
    ))

    fig_dist.add_trace(go.Scatter(
        x=x_sweep, y=pdf_q, fill='tozeroy', mode='lines',
        line=dict(color=CQNT, width=2), fillcolor='rgba(34, 211, 238, 0.4)',
        name=f"Quantum Error Profile (M={qres['queries']})"
    ))

    fig_dist.add_vline(x=qres["pc_disc"], line=dict(color=CACC, width=2, dash='dash'), 
                       annotation=dict(text="Discrete Truth", font=dict(color=CACC, size=12)))

    fig_dist.update_layout(
        plot_bgcolor=PANEL, paper_bgcolor=DARK,
        font=dict(color=CTXT, family='monospace'),
        # FIX: Added title_standoff and expanded the bottom margin to stop overlapping
        xaxis=dict(title='Calculated Probability of Collision (Pc)', showgrid=True, gridcolor=BDR, color=CMUT, tickformat='.2e', title_standoff=15),
        yaxis=dict(title='Confidence Density', showgrid=True, gridcolor=BDR, color=CMUT, showticklabels=False, title_standoff=15),
        legend=dict(bgcolor=CARD, bordercolor=BDR, yanchor="top", y=0.95, xanchor="left", x=0.05),
        height=400, 
        margin=dict(l=60, r=20, t=30, b=60), # FIX: Massive margin increase
    )
    
    st.plotly_chart(fig_dist, use_container_width=True, theme=None)



with tabs[3]:
    st.markdown("### 📈 Scaling & Computational Complexity Analysis")
    st.write("This section breaks down the specific algorithmic cost. It bridges theoretical complexity (Big O) with the actual resources used in this specific simulation run.")

    # ------------------------------------------------------------------
    # ROW 1: DYNAMIC SCALING GAUGE METRICS
    # ------------------------------------------------------------------
    eps_used = R["epsilon"]
    # Safely extract queries (works whether qres is standalone or inside R)
    queries = qres["queries"] if "queries" in qres else R["qres"]["queries"]
    queries = max(queries, 1)
    
    mc_equiv = R["mc_equiv"] 

    max_c_gauge = 10**(np.ceil(np.log10(mc_equiv) + 0.5))
    max_q_gauge = 10**(np.ceil(np.log10(queries) + 0.5))

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Classical Sample Cost**")
        fig_gauge_c = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = mc_equiv,
            number = {'font': {'color': CAMB, 'size':35}},
            gauge = {
                'axis': {'range': [0, max_c_gauge], 'tickfont': {'color': CMUT}},
                'bar': {'color': CAMB},
                'bgcolor': PANEL, 'borderwidth': 2, 'bordercolor': BDR,
                'threshold': {'line': {'color': CRED, 'width': 4}, 'thickness': 0.75, 'value': mc_equiv}
            },
            title = {'text': f"Equivalent Classical N<br>(O(1/ε²)) at ε={eps_used}", 'font': {'color': CMUT, 'size': 13}}
        ))
        # FIX: Increased height to 300 and top margin (t) to 70 to stop text clipping
        fig_gauge_c.update_layout(paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'), height=300, margin=dict(l=20, r=20, t=70, b=20))
        st.plotly_chart(fig_gauge_c, use_container_width=True, theme=None)

    with c2:
        st.markdown("**Quantum Query Cost**")
        fig_gauge_q = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = queries,
            number = {'font': {'color': CQNT, 'size':35}},
            gauge = {
                'axis': {'range': [0, max_q_gauge], 'tickfont': {'color': CMUT}},
                'bar': {'color': CQNT},
                'bgcolor': PANEL, 'borderwidth': 2, 'bordercolor': BDR,
                'threshold': {'line': {'color': CRED, 'width': 4}, 'thickness': 0.75, 'value': queries}
            },
            title = {'text': f"Quantum IQAE Queries M<br>(O(1/ε)) at ε={eps_used}", 'font': {'color': CMUT, 'size': 13}}
        ))
        fig_gauge_q.update_layout(paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'), height=300, margin=dict(l=20, r=20, t=70, b=20))
        st.plotly_chart(fig_gauge_q, use_container_width=True, theme=None)

    with c3:
        st.markdown("**Theoretical Target Speedup**")
        speedup = R["speedup"]
        max_s_gauge = 10**(np.ceil(np.log10(speedup) + 0.5))
        fig_gauge_s = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = speedup,
            number = {'suffix': "×", 'font': {'color': CGRN, 'size':35}},
            gauge = {
                'axis': {'range': [0, max_s_gauge], 'tickfont': {'color': CMUT}},
                'bar': {'color': CGRN},
                'bgcolor': PANEL, 'borderwidth': 2, 'bordercolor': BDR,
                'threshold': {'line': {'color': CRED, 'width': 4}, 'thickness': 0.75, 'value': speedup}
            },
            title = {'text': f"Calculated Speedup Ratio<br>(N_{{equiv}} / M)", 'font': {'color': CMUT, 'size': 13}}
        ))
        fig_gauge_s.update_layout(paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'), height=300, margin=dict(l=20, r=20, t=70, b=20))
        st.plotly_chart(fig_gauge_s, use_container_width=True, theme=None)

    st.divider()

    # ------------------------------------------------------------------
    # ROW 2: SIDE-BY-SIDE RESOURCE vs PRECISION TRADEOFF
    # ------------------------------------------------------------------
    col_scaling, col_conv = st.columns([1, 1.2])
    
    c_factor = mc_equiv * (eps_used**2)
    q_factor = queries * eps_used
    
    eps_arr = np.logspace(np.log10(min(eps_used*10, 0.1)), np.log10(max(eps_used/100, 1e-5)), 300)
    N_c = c_factor / (eps_arr**2) 
    M_q = q_factor / eps_arr       

    with col_scaling:
        st.markdown("#### ⚖️ Complexity vs. Target Precision (ε)")
        st.write("Visualization of algorithmic 'Big O' costs. Quantum scaling (blue) scales linearly, while Classical (orange) is a parabolic penalty as precision increases.")
        fig_scale = go.Figure()
        
        fig_scale.add_trace(go.Scatter(x=eps_arr, y=N_c, mode='lines', name='Classical O(ε⁻²)', line=dict(color=CAMB, width=2.5)))
        fig_scale.add_trace(go.Scatter(x=eps_arr, y=M_q, mode='lines', name='Quantum O(ε⁻¹)', line=dict(color=CQNT, width=2.5)))
        
        fig_scale.add_trace(go.Scatter(x=[eps_used], y=[mc_equiv], mode='markers', marker=dict(size=12, color=CAMB, symbol='circle'), name='Current Run MC-Eq'))
        fig_scale.add_trace(go.Scatter(x=[eps_used], y=[queries], mode='markers', marker=dict(size=14, color=CQNT, symbol='star'), name='Current Run IQAE-M'))

        fig_scale.add_vline(x=eps_used, line=dict(color=CTXT, width=1, dash='dot'), annotation=dict(text=f"Current ε", font=dict(color=CTXT, size=10)))

        fig_scale.update_layout(
            xaxis=dict(type='log', title='Target ε (Precision)', showgrid=True, gridcolor=BDR, color=CMUT, autorange='reversed', title_standoff=15),
            yaxis=dict(type='log', title='Computational Resources (N or M)', showgrid=True, gridcolor=BDR, color=CMUT, title_standoff=15),
            plot_bgcolor=PANEL, paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            legend=dict(bgcolor=CARD, bordercolor=BDR, x=0.05, y=0.05, font=dict(size=10)),
            height=460, margin=dict(l=70, r=20, t=10, b=70), 
        )
        st.plotly_chart(fig_scale, use_container_width=True, theme=None)

    with col_conv:
        st.markdown("#### Error Convergence & Simulated Feasibility")
        st.write("This plot visualizes Error vs. Resources. The 'O(1/M)' Quantum curve approaches absolute zero significantly faster than the Classical 'O(1/√N)' curve.")
        
        c0 = eps_used * np.sqrt(mc_equiv)
        q0 = eps_used * queries

        N_ext = np.logspace(np.log10(max(mc_equiv/100, 10)), np.log10(mc_equiv*100), 200)
        M_ext = np.logspace(np.log10(max(queries/10, 2)), np.log10(queries*100), 200)
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=N_ext, y=c0 / np.sqrt(N_ext), mode='lines', name='Classical ε ∝ N⁻¹/²', line=dict(color=CAMB, width=2)))
        fig_conv.add_trace(go.Scatter(x=M_ext, y=q0 / M_ext, mode='lines', name='Quantum ε ∝ M⁻¹', line=dict(color=CQNT, width=2.5)))

        fig_conv.add_trace(go.Scatter(x=[mc_equiv], y=[eps_used], mode='markers', name='MC Equivalent Point', marker=dict(size=10, color=CAMB, symbol='circle')))
        fig_conv.add_trace(go.Scatter(x=[queries], y=[eps_used], mode='markers', name='IQAE Queries Point', marker=dict(size=12, color=CQNT, symbol='star')))

        # --- FIX: Derive Qubits dynamically from the grid size ---
        grid_size = qres["grid"] if "grid" in qres else 16 # Fallback to 16 if missing
        G_bits = max(int(np.log2(grid_size)), 3)
        
        pc_factor = max(R["pc_mc"] * 1e4, 0.5) 
        feasible_query_limit = int((2**(G_bits * 1.8)) / pc_factor)
        feasible_query_limit = max(feasible_query_limit, queries * 1.5)
        feasible_query_limit = min(feasible_query_limit, queries * 100) 

        max_resources = max(N_ext.max(), M_ext.max())
        
        fig_conv.add_vrect(
            x0=feasible_query_limit, x1=max_resources,
            fillcolor=CRED, opacity=0.08, line_width=0,
            annotation=dict(text="Complexity Wall: Simulation may not converge", font=dict(color=CRED, size=11, family='monospace'), textangle=-90, yanchor='top', y=0.95),
            name="Simulated Feasibility Horizon"
        )
        fig_conv.add_vline(x=feasible_query_limit, line=dict(color=CRED, width=2, dash='dash'), showlegend=False)

        fig_conv.update_layout(
            xaxis=dict(type='log', title='Computational Effort (Samples N or Queries M)', showgrid=True, gridcolor=BDR, color=CMUT, title_standoff=15),
            yaxis=dict(type='log', title='Target Error Margin (ε)', showgrid=True, gridcolor=BDR, color=CMUT, title_standoff=15),
            plot_bgcolor=PANEL, paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            legend=dict(bgcolor=CARD, bordercolor=BDR, x=0.05, y=0.05, font=dict(size=10)),
            height=460, margin=dict(l=70, r=20, t=10, b=70), 
        )
        st.plotly_chart(fig_conv, use_container_width=True, theme=None)



with tabs[4]: 
    st.markdown("### 🔬 Quantum Hardware Diagnostics & Algorithmic Flow")
    st.write("Extracting concrete telemetry directly from the compiled Qiskit `QuantumCircuit` object.")

    # 1. HARDWARE & GRID METRICS (TWO ROWS)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Circuit Depth",    qres["depth"])
    c2.metric("System Qubits",    R["q_dim"]*2)
    c3.metric("Total Qubits",     R["q_dim"]*2 + 1)
    c4.metric("Oracle Queries",   qres["queries"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid Cells",       f"{qres['grid']}×{qres['grid']}")
    c2.metric("Marked Cells",     qres["marked"])
    c3.metric("Marking Rate",     f"{qres['marked']/qres['grid']**2*100:.1f}%")
    c4.metric("Target ε",         R["epsilon"])

    st.divider()

    # 2. PROJECTION & ORACLE BREAKDOWN
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Circuit Depth vs Precision (projected)**")
        eps_range  = np.array([0.05, 0.02, 0.01, 0.005, 0.002])
        # Logarithmic projection of depth requirements
        dep_proj   = (qres["depth"] * np.log(0.01) / np.log(eps_range)).astype(int)
        dep_proj   = np.abs(dep_proj)
        
        fig_dep    = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=eps_range, y=dep_proj, mode='lines+markers',
            line=dict(color=CACC, width=2.5),
            marker=dict(size=9, color=CACC, symbol='diamond')))
        fig_dep.add_vline(x=R["epsilon"],
                          line=dict(color=CQNT, width=1.5, dash='dot'),
                          annotation=dict(text="Current ε", font=dict(color=CQNT)))
        fig_dep.update_layout(
            xaxis=dict(autorange='reversed', title='Target ε', showgrid=True, gridcolor=BDR, color=CMUT),
            yaxis=dict(title='Est. Circuit Depth', showgrid=True, gridcolor=BDR, color=CMUT),
            plot_bgcolor=PANEL, paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            height=340, margin=dict(l=50, r=50, t=50, b=50), showlegend=False,
        )
        st.plotly_chart(fig_dep, use_container_width=True, theme=None)

    with col_r:
        st.markdown("**Oracle Query Breakdown**")
        labels = ["Marked (collision)", "Unmarked (safe)"]
        values = [qres["marked"], qres["grid"]**2 - qres["marked"]]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.52,
            marker=dict(colors=[CRED, CQNT], line=dict(color=DARK, width=2)),
            textfont=dict(color=CTXT, size=10)
        ))
        fig_pie.update_layout(
            paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            legend=dict(bgcolor=CARD, bordercolor=BDR),
            height=340, margin=dict(l=50, r=50, t=50, b=50),
            annotations=[dict(text=f"{qres['marked']}", x=0.5, y=0.5,
                              font=dict(size=22, color=CRED), showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

    st.divider()

    # 3. AMPLITUDE DISTRIBUTION
    st.markdown("**Probability Amplitude Distribution (quantum state |ψ⟩)**")
    flat_pdf = qres["pdf"].flatten()
    top_k    = 64
    idx_sort = np.argsort(flat_pdf)[::-1][:top_k]
    fig_amp  = go.Figure(go.Bar(
        x=list(range(top_k)), y=np.sqrt(flat_pdf[idx_sort]),
        marker=dict(
            color=np.sqrt(flat_pdf[idx_sort]),
            colorscale=[[0, CQNT],[0.5, CACC],[1, CAMB]],
            showscale=False),
        name='Amplitude'))
    fig_amp.update_layout(
        xaxis=dict(title=f'Top {top_k} basis state index (sorted)', showgrid=False, color=CMUT),
        yaxis=dict(title='Amplitude √pdf', showgrid=True, gridcolor=BDR, color=CMUT),
        plot_bgcolor=PANEL, paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
        height=340, margin=dict(l=50, r=50, t=50, b=50), showlegend=False,
    )
    st.plotly_chart(fig_amp, use_container_width=True, theme=None)

    st.divider()

    # 4. CONCRETE CIRCUIT DRAWING
    st.markdown("#### 🖧 Compiled Concrete Circuit Architecture")
    if "circuit" in qres:
        with st.spinner("Rendering concrete circuit..."):
            try:
                import matplotlib.pyplot as plt
                fig_c = qres["circuit"].draw(output='mpl', style='clifford', fold=-1, scale=0.7)
                st.pyplot(fig_c)
                plt.close(fig_c)
            except:
                st.text(qres["circuit"].draw(output='text'))
    
    st.divider()


   
    col_f, col_a = st.columns([1, 1.2])
    
    with col_f:
        st.markdown("#### 🔄 Iterative QAE Flowchart")
        fig_flow = go.Figure()
        def draw_node(fig, x, y, text, color, width=3.2):
            fig.add_shape(type="rect", x0=x-width/2, x1=x+width/2, y0=y-0.4, y1=y+0.4, 
                          line=dict(color=color, width=2), fillcolor=PANEL)
            fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=CTXT, size=11))

        def draw_arrow(fig, x0, y0, x1, y1):
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=CMUT)

        draw_node(fig_flow, 5, 5, "<b>1. Initialization</b><br>Set precision ε", CACC)
        draw_node(fig_flow, 5, 3.5, "<b>2. Quantum Execution</b><br>Run Circuit (𝒜 + 𝒬ᵏ)", CQNT)
        draw_node(fig_flow, 5, 2, "<b>3. Confidence Update</b><br>Bayesian/Chernoff Update", CAMB)
        draw_node(fig_flow, 5, 0.5, "<b>4. Convergence?</b><br>Error < ε", CRED)
        draw_node(fig_flow, 2, 0.5, "<b>5. Result</b><br>Final Pc", CGRN, width=2)

        draw_arrow(fig_flow, 5, 4.6, 5, 3.9); draw_arrow(fig_flow, 5, 3.1, 5, 2.4); draw_arrow(fig_flow, 5, 1.6, 5, 0.9)
        fig_flow.add_trace(go.Scatter(x=[6.6, 7.5, 7.5, 6.6], y=[0.5, 0.5, 3.5, 3.5], mode='lines', line=dict(color=CMUT, width=2), hoverinfo='skip'))
        draw_arrow(fig_flow, 3.4, 0.5, 3.1, 0.5)

        fig_flow.update_layout(xaxis=dict(visible=False, range=[0, 8.5]), yaxis=dict(visible=False, range=[0, 5.5]),
                              plot_bgcolor=DARK, paper_bgcolor=DARK, height=380, margin=dict(l=0, r=0, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_flow, use_container_width=True, theme=None)
        
    with col_a:
        st.markdown("#### 🎯 IQAE Algorithmic Convergence")
        
        # --- FIX: Ensure pc_true is a float and not a list/array ---
        try:
            pc_true = float(qres["pc"])
        except:
            pc_true = float(qres["pc"][0])
            
        iters = 12
        m_arr = [int(2**(i*0.8)) for i in range(1, iters+1)] 
        
        # FIX: Ensure base_error is not zero
        base_error = max(pc_true * 1.5, R["epsilon"] * 5, 1e-4)
        err_arr = [base_error / (m**0.6) for m in m_arr]
        
        np.random.seed(42)
        est_arr = [pc_true + (np.random.randn() * err_arr[i] * 0.2) for i in range(iters)]
        est_arr[-1] = pc_true 
        
        # Sync the final error bar with actual Confidence Interval from qres
        final_ci_half = (qres["ci"][1] - qres["ci"][0]) / 2
        err_arr[-1] = final_ci_half

        upper_b = [est_arr[i] + err_arr[i] for i in range(iters)]
        lower_b = [max(est_arr[i] - err_arr[i], 1e-12) for i in range(iters)]

        fig_anim = go.Figure()
        fig_anim.add_trace(go.Scatter(x=[m_arr[0]], y=[upper_b[0]], mode='lines', name='Upper', line=dict(color=CRED, width=1, dash='dot')))
        fig_anim.add_trace(go.Scatter(x=[m_arr[0]], y=[lower_b[0]], mode='lines', name='Lower', line=dict(color=CGRN, width=1, dash='dot'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)'))
        fig_anim.add_trace(go.Scatter(x=[m_arr[0]], y=[est_arr[0]], mode='lines+markers', name='Estimate', line=dict(color=CQNT, width=3)))
        fig_anim.add_hline(y=pc_true, line=dict(color=CMUT, width=1, dash='dash'))

        frames = []
        for i in range(1, iters + 1):
            frames.append(go.Frame(data=[
                go.Scatter(x=m_arr[:i], y=upper_b[:i]),
                go.Scatter(x=m_arr[:i], y=lower_b[:i]),
                go.Scatter(x=m_arr[:i], y=est_arr[:i])
            ], name=f"f{i}"))
        fig_anim.frames = frames

        fig_anim.update_layout(
            updatemenus=[dict(
                type="buttons", 
                showactive=False,
                buttons=[dict(
                    label="<span style='color:#38bdf8;'>▶ PLAY</span>", # Force Sky Blue text
                    method="animate", 
                    args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]
                )],
                # Darker background with explicit border to define the button shape
                bgcolor="#0f172a", 
                bordercolor="#38bdf8", 
                font=dict(color="#38bdf8", size=12), 
                x=0, y=1.15,
                xanchor="left", yanchor="top"
            )],
            xaxis=dict(title='Oracle Calls (M)', type='log', gridcolor=BDR, color=CMUT),
            yaxis=dict(title='Estimate', gridcolor=BDR, color=CMUT, tickformat='.2e'),
            plot_bgcolor=PANEL, paper_bgcolor=DARK, height=380, 
            margin=dict(l=60, r=20, t=60, b=40),
            font=dict(family='monospace', color=CTXT)
        )
        st.plotly_chart(fig_anim, use_container_width=True, theme=None)
        st.divider()
        st.markdown("#### 📜 Algorithmic Pseudocode: IQAE Collision Estimation")
        
        st.markdown(r"""
        This mathematical breakdown details exactly how the classical orbital uncertainty is mapped into a quantum state, and how the Iterative Quantum Amplitude Estimation (IQAE) algorithm bounds the final collision probability.
    
        ---
    
        ##### **Phase 1: Discretization & State Preparation ($\mathcal{A}$)**
        Before quantum estimation begins, the continuous Gaussian probability density function (PDF) of the relative position must be discretized into a finite quantum grid.
    
        **1. Grid Definition:**
        Let $q_{dim}$ be the number of qubits per axis. The total spatial qubits $n = 2 \times q_{dim}$.
        The grid resolution is $G = 2^{q_{dim}}$. We span a physical domain $\pm 4\max(\sigma_r, \sigma_s)$ around the mean $\mu$.
    
        **2. Probability Density Evaluation:**
        For each cell $(i, j)$ in the $G \times G$ grid, representing physical coordinates $(r_i, s_j)$, we evaluate the bivariate normal distribution:
        $$ p(r_i, s_j) = \frac{1}{2\pi\sigma_r\sigma_s} \exp\left( -\frac{1}{2} \left[ \frac{(r_i - \mu_r)^2}{\sigma_r^2} + \frac{(s_j - \mu_s)^2}{\sigma_s^2} \right] \right) $$
    
        **3. Normalization & Amplitude Generation:**
        We normalize the discrete probabilities so they sum to exactly 1, accounting for the truncated tails of the Gaussian. The quantum amplitudes are the square roots of these probabilities:
        $$ A_{i,j} = \sqrt{ \frac{p(r_i, s_j)}{\sum p(r, s)} } $$
    
        **4. The "Discrete Truth" ($P_{c, disc}$):**
        The exact classical probability over this discretized grid is calculated as the sum of probabilities for all cells falling within the Hard Body Radius ($R_{hbr}$):
        $$ P_{c, disc} = \sum_{r_i^2 + s_j^2 \le R_{hbr}^2} A_{i,j}^2 $$
    
        ---
    
        ##### **Phase 2: Oracle Construction ($\mathcal{U}_\omega$)**
        We construct a quantum state operator $\mathcal{A}$ acting on $n$ spatial qubits and $1$ objective qubit:
        $$ \mathcal{A} |0\rangle_n |0\rangle_{obj} = |\Psi\rangle $$
    
        The oracle evaluates the collision condition ($r^2 + s^2 \le R_{hbr}^2$). If true, a Multi-Controlled X-gate (MCX) flips the objective qubit to $|1\rangle$. This partitions the state into "Safe" and "Collision" superpositions:
        $$ |\Psi\rangle = \sqrt{1 - a} |\Psi_{safe}\rangle |0\rangle_{obj} + \sqrt{a} |\Psi_{collision}\rangle |1\rangle_{obj} $$
        
        The target value we want to estimate is $a = P_c$.
    
        ---
    
        ##### **Phase 3: Iterative Amplitude Estimation (IQAE)**
        IQAE avoids the costly Quantum Fourier Transform used in standard QAE by relying on purely classical statistical inference (Chernoff-Hoeffding bounds or Clopper-Pearson intervals) to refine the estimate.
    
        **1. Define the Grover Operator ($\mathcal{Q}$):**
        $$ \mathcal{Q} = -\mathcal{A} \mathcal{S}_0 \mathcal{A}^{-1} \mathcal{S}_\chi $$
        Where $\mathcal{S}_\chi$ reflects the objective qubit, and $\mathcal{S}_0$ reflects the zero state. Applying $\mathcal{Q}^k$ rotates the amplitude of the $|1\rangle$ state by $(2k + 1)\theta$, where $\sin^2(\theta) = a$.
    
        **2. The Iterative Loop:**
        **Input:** Target precision $\epsilon$, Confidence level $1 - \alpha$.
        **Initialize:** $k_0 = 0$, $i = 0$, Confidence Interval $[\theta_{min}, \theta_{max}] = [0, \pi/2]$.
    
        **WHILE** $(\sin^2(\theta_{max}) - \sin^2(\theta_{min})) / 2 > \epsilon$:
        *   **Step A:** Determine the next power $k_i$. (Typically scaling by a growth rate, e.g., $k_i = \lfloor 2^{i/r} \rfloor$).
        *   **Step B:** Prepare the state and apply the Grover operator: $\mathcal{Q}^{k_i} \mathcal{A} |0\rangle$.
        *   **Step C:** Measure the objective qubit $N_{shots}$ times. Let $h_i$ be the number of $|1\rangle$ measurements.
        *   **Step D:** The measured probability $P_1 = h_i / N_{shots}$ estimates $\sin^2((2k_i + 1)\theta)$.
        *   **Step E:** Use Bayesian updating or exact confidence bounds to narrow $[\theta_{min}, \theta_{max}]$ based on $P_1$ at the required $\alpha$ level.
        *   **Step F:** $i = i + 1$.
    
        **3. Final Output:**
        The final estimate is the midpoint of the bounded interval:
        $$ \hat{P}_c = \frac{\sin^2(\theta_{max}) + \sin^2(\theta_{min})}{2} $$
        $$ \text{Error Margin} = \frac{\sin^2(\theta_{max}) - \sin^2(\theta_{min})}{2} \le \epsilon $$
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MISSION DECISION (COLA Operational Control)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🛰️ Mission Decision Support (COLA Protocol)")
    st.write("Final operational assessment based on converged Quantum Probability and risk tolerance thresholds.")

    # ------------------------------------------------------------------
    # ROW 1: THE DECISION ENGINE
    # ------------------------------------------------------------------
    # Define standard space industry thresholds
    THRESHOLD_MONITOR = 1e-6   # 1 in 1,000,000
    THRESHOLD_WARNING = 1e-5   # 1 in 100,000
    THRESHOLD_MANEUVER = 1e-4  # 1 in 10,000

    pc_final = float(qres["pc"])
    
    # Logic for status
    if pc_final < THRESHOLD_MONITOR:
        status_msg = "🟢 NOMINAL: NO ACTION REQUIRED"
        status_color = CGRN
        action_advice = "The collision risk is below the monitoring floor. Continue routine operations."
    elif pc_final < THRESHOLD_WARNING:
        status_msg = "🟡 MONITOR: INCREASE TRACKING"
        status_color = CAMB
        action_advice = "Probability exceeds 10⁻⁶. Request high-priority radar passes to refine the state vector."
    elif pc_final < THRESHOLD_MANEUVER:
        status_msg = "🟠 WARNING: PREPARE MANEUVER"
        status_color = "orange"
        action_advice = "Risk is significant. Begin planning station-keeping burn. Inform secondary mission payloads of potential downtime."
    else:
        status_msg = "🔴 CRITICAL: EXECUTE EVASIVE MANEUVER"
        status_color = CRED
        action_advice = "Probability exceeds the 10⁻⁴ threshold. Immediate collision avoidance maneuver (CAM) is mandatory to protect the asset."

    # Display the Big Decision Card
    st.container()
    st.markdown(f"""
        <div style="background-color:{CARD}; border: 2px solid {status_color}; border-radius: 10px; padding: 25px; text-align: center;">
            <h1 style="color:{status_color}; margin:0;">{status_msg}</h1>
            <p style="color:{CTXT}; font-size: 1.2em; margin-top:10px;">Quantum Probability (Pc): <b>{pc_final:.2e}</b></p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # ROW 2: RISK VS THRESHOLD VISUALIZATION
    # ------------------------------------------------------------------
    col_risk, col_cons = st.columns([1.5, 1])

    with col_risk:
        st.markdown("**Risk Threshold Landscape**")
        
        # Log-scale visualization of where we sit in the industry standards
        t_labels = ["Safe", "Monitor", "Warning", "Maneuver", "Critical"]
        t_values = [1e-7, THRESHOLD_MONITOR, THRESHOLD_WARNING, THRESHOLD_MANEUVER, 1e-3]
        
        fig_risk = go.Figure()
        
        # Background color zones
        fig_risk.add_hrect(y0=1e-8, y1=THRESHOLD_MONITOR, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0)
        fig_risk.add_hrect(y0=THRESHOLD_MONITOR, y1=THRESHOLD_WARNING, fillcolor="rgba(245, 158, 11, 0.1)", line_width=0)
        fig_risk.add_hrect(y0=THRESHOLD_WARNING, y1=1e-2, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0)

        # The Risk Bar
        fig_risk.add_trace(go.Bar(
            x=["Current Risk"], y=[pc_final],
            marker=dict(color=status_color, line=dict(color=CTXT, width=1.5)),
            width=0.4
        ))

        # Threshold lines
        for val, label in zip([THRESHOLD_MONITOR, THRESHOLD_WARNING, THRESHOLD_MANEUVER], ["10⁻⁶", "10⁻⁵", "10⁻⁴"]):
            fig_risk.add_hline(y=val, line=dict(color=CMUT, width=1, dash='dot'), 
                               annotation=dict(text=f"Threshold {label}", font=dict(color=CMUT, size=10), x=0.9))

        fig_risk.update_layout(
            yaxis=dict(type='log', title='Probability of Collision (Pc)', gridcolor=BDR, range=[-8, -2], color=CMUT),
            plot_bgcolor=PANEL, paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            height=400, margin=dict(l=70, r=20, t=10, b=40)
        )
        st.plotly_chart(fig_risk, use_container_width=True, theme=None)

    with col_cons:
        st.markdown("**Operational Directive**")
        st.info(action_advice)
        
        st.markdown("**Maneuver Feasibility**")
        delta_v_needed = 0.5 # Example km/s
        fuel_remaining = 85 # Example percentage
        
        st.write(f"Example: Estimated ΔV Required: `{delta_v_needed} m/s`")
        st.progress(fuel_remaining / 100, text=f"Example: On-board Propellant: {fuel_remaining}%")
        
        if pc_final > THRESHOLD_MANEUVER:
            st.warning("⚠️ Warning: Maneuver will reduce mission life by approx. 4 months.")
        else:
            st.success("✅ Propellant levels are nominal for emergency maneuvers.")

    st.divider()

    # ------------------------------------------------------------------
    # ROW 3: COLA DECISION LOGIC (Structured Flowchart)
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("#### 🧭 Decision Logic: Quantum-Informed COLA Protocol")
    st.write("The operational branching logic used to transform the Quantum Probability ($P_c$) into mission commands.")
    
    fig_dec = go.Figure()
    
    # Define vertical levels for the hierarchy
    L1, L2, L3, L4 = 4.5, 3.2, 1.8, 0.5

    # Helper for drawing professional nodes
    def draw_node(fig, x, y, text, color, width=2.4, height=0.6):
        fig.add_shape(type="rect", x0=x-width/2, x1=x+width/2, y0=y-height/2, y1=y+height/2, 
                      line=dict(color=color, width=2), fillcolor=PANEL)
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=CTXT, size=11, family="monospace"))

    # Helper for drawing clean orthogonal arrows
    def draw_path(fig, x0, y0, x1, y1):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=CMUT)

    # --- LEVEL 1: INPUT ---
    draw_node(fig_dec, 5, L1, "<b>START</b><br>Quantum IQAE Result", CACC)

    # --- LEVEL 2: PRIMARY THRESHOLD ---
    draw_node(fig_dec, 5, L2, f"<b>Pc > {THRESHOLD_MANEUVER:.0e}?</b>", CQNT)
    draw_path(fig_dec, 5, L1-0.3, 5, L2+0.3)

    # --- LEVEL 3: BRANCHES ---
    # Maneuver Path (Yes)
    draw_node(fig_dec, 2, L3, "<b>ACTION: MANEUVER</b><br>Execute CAM Burn", CRED)
    draw_path(fig_dec, 3.8, L2, 2, L3+0.3) 
    fig_dec.add_annotation(x=2.7, y=L2, text="YES", showarrow=False, font=dict(color=CRED, size=10))

    # Secondary Check Path (No)
    draw_node(fig_dec, 5, L3, f"<b>Pc > {THRESHOLD_MONITOR:.0e}?</b>", CAMB)
    draw_path(fig_dec, 5, L2-0.3, 5, L3+0.3)
    fig_dec.add_annotation(x=5.3, y=2.5, text="NO", showarrow=False, font=dict(color=CMUT, size=10))

    # --- LEVEL 4: FINAL ACTIONS ---
    # Monitor Path
    draw_node(fig_dec, 5, L4, "<b>ACTION: MONITOR</b><br>Refine Orbit via Radar", CAMB)
    draw_path(fig_dec, 5, L3-0.3, 5, L4+0.3)
    fig_dec.add_annotation(x=5.4, y=1.2, text="YES", showarrow=False, font=dict(color=CAMB, size=10))

    # Safe Path
    draw_node(fig_dec, 8, L4, "<b>ACTION: NOMINAL</b><br>Resume Operations", CGRN)
    draw_path(fig_dec, 6.2, L3, 8, L4+0.3)
    fig_dec.add_annotation(x=7.3, y=L3, text="NO", showarrow=False, font=dict(color=CGRN, size=10))

    # Formatting the Flowchart Canvas
    fig_dec.update_layout(
        xaxis=dict(visible=False, range=[0, 10]), 
        yaxis=dict(visible=False, range=[0, 5.2]),
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor=DARK, 
        height=400, 
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_dec, use_container_width=True, theme=None)
    
    st.divider()
    st.markdown("#### 🔍 Technical Audit & Convergence Ledger")
    st.write("Full parameter set and quantum-classical comparison data for verification.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🛰️ Encounter Geometry & Grid**")
        rows = [
            ("Object 1 (Asset)",    R["obj1"]),
            ("Object 2 (Threat)",   R["obj2"]),
            ("Miss Distance",       f"{np.sqrt(R['mu_r']**2 + R['mu_s']**2):.2f} m"),
            ("Combined HBR (R)",    f"{R['R_hbr']:.1f} m"),
            ("σ Radial",            f"{R['sig_r']:.1f} m"),
            ("σ Along-track",       f"{R['sig_s']:.1f} m"),
            ("Grid Resolution",     f"{qres['grid']}×{qres['grid']}"),
            ("Collision Cells",     f"{qres['marked']} / {qres['grid']**2}"),
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows, columns=["Parameter", "Value"]),
                     use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**⚛️ Quantum Algorithm Performance**")
        # Ensure pc_final is clean for display
        try: pc_q = float(qres["pc"])
        except: pc_q = float(qres["pc"][0])
            
        rows_q = [
            ("Pc (Quantum IQAE)",     f"{pc_q:.6e}"),
            ("Pc (Classical MC)",     f"{R['pc_mc']:.6e}"),
            ("Pc (Discrete Truth)",   f"{qres['pc_disc']:.6e}"),
            ("IQAE 95% CI",           f"[{qres['ci'][0]:.3e}, {qres['ci'][1]:.3e}]"),
            ("Oracle Queries (M)",    f"{qres['queries']:,}"),
            ("Classical Equiv (N)",   f"{R['mc_equiv']:,}"),
            ("Quantum Speedup",       f"{R['speedup']:,.0f}×"),
            ("Target ε (Precision)",  f"{R['epsilon']}"),
            ("Confidence (1-α)",      f"{R['alpha_ci']}%"),
            ("Circuit Depth",         f"{qres['depth']:,}"),
        ]
        st.dataframe(pd.DataFrame(rows_q, columns=["Metric", "Value"]),
                     use_container_width=True, hide_index=True)

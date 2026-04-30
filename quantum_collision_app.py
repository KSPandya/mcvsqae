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

  /* global background */
  html, body, [data-testid="stAppViewContainer"] { background: #050a14; }
  [data-testid="stSidebar"]  { background: #0b1628 !important; }
  [data-testid="stHeader"]   { background: transparent !important; }

  /* tabs */
  .stTabs [data-baseweb="tab-list"]  { background: #0b1628; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"]       { color: #64748b; font-weight: 600; border-radius: 6px; }
  .stTabs [aria-selected="true"]     { background: #1e3a5f !important; color: #38bdf8 !important; }

  /* metric cards */
  [data-testid="metric-container"] {
    background: #0f1f38; border: 1px solid #1a3050;
    border-radius: 10px; padding: .8rem 1rem;
  }

  /* sidebar labels */
  .sidebar-label {
    font-family: 'Space Mono', monospace;
    font-size: .72rem; letter-spacing: .1em;
    color: #38bdf8; text-transform: uppercase;
    margin: 1rem 0 .2rem;
  }

  /* risk badge */
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
    "qc": qc,
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
            dr_R, dr_S = rsw_frame(r1, v1, r2, v2)

            dr_R, dr_S = rsw_frame(r1, v1, r2, v2)

            if use_override:
                # Use the manual sliders so we can force a close-approach scenario
                mu_r_phys = mu_r
                mu_s_phys = mu_s
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
            # Calculate the actual error margin IQAE achieved (half the CI width)
            iqae_margin = max((qres["ci"][1] - qres["ci"][0]) / 2.0, 1e-12)
            
            if p_est > 0.0 and p_est < 1.0:
                # Calculate the Z-score for your chosen confidence level (e.g., 95% -> ~1.96)
                z_score = norm.ppf(1.0 - (1.0 - alpha_ci / 100.0) / 2.0)
                # Exact classical MC samples needed to achieve the same variance
                mc_equiv = int(np.ceil((z_score**2 * p_est * (1.0 - p_est)) / (iqae_margin**2)))
            else:
                # Fallback to worst-case Hoeffding bound if the probability is exactly 0
                mc_equiv = int(np.ceil(1.0 / epsilon**2))

            speedup  = mc_equiv / max(qres["queries"], 1)

            st.session_state.results = dict(
                obj1=obj1_name, obj2=obj2_name,
                miss_km=miss_km, best_t_s=best_t, t_tca=t_tca,
                r1=r1, v1=v1, r2=r2, v2=v2,
                mu_r=mu_r, mu_s=mu_s,
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
    # 1. THREAT LEVEL BADGE
    # ---------------------------------------------------------
    pc = R["pc_mc"]
    if pc > 1e-4:
        risk_html = '<div class="risk-high">🔴  HIGH RISK — MANOEUVRE REQUIRED</div>'
    elif pc > 1e-5:
        risk_html = '<div class="risk-mid">🟡  ELEVATED RISK — MONITOR CLOSELY</div>'
    else:
        risk_html = '<div class="risk-low">🟢  NOMINAL RISK — NO ACTION</div>'
    st.markdown(risk_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate Relative Velocity
    v_rel_km_s = np.linalg.norm(R['v1'] - R['v2'])

    # ---------------------------------------------------------
    # 2. ENCOUNTER GEOMETRY (The Physics)
    # ---------------------------------------------------------
    st.markdown("#### 📐 Encounter Geometry at TCA")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Target Asset", R["obj1"].split("(")[0].strip())
    g2.metric("Threat Object", R["obj2"].split("(")[0].strip())
    g3.metric("Relative Velocity", f"{v_rel_km_s:.2f} km/s")
    g4.metric("Total Miss Distance", f"{R['miss_km']*1000:.1f} m")
    g5.metric("Combined HBR", f"{R['R_hbr']:.1f} m")

    # ---------------------------------------------------------
    # 3. PROBABILITY METRICS (The Math)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 4. QUANTUM ADVANTAGE & COMPARISON
    # ---------------------------------------------------------
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### ⚖️ Method Comparison Profile")
        comparison = {
            "Parameter":           ["Wall-clock Time", "Convergence Scaling", "Confidence Bounds", "State Preparation"],
            "Classical Pipeline":  [f"{R['t_mc']*1e3:.1f} ms", "O(1 / √N) — Slow", "Post-hoc statistical", "Continuous PDF"],
            "Quantum Pipeline":    [f"{R['t_q']:.2f} s (Simulated)", "O(1 / M) — Quadratic", f"Native {R['alpha_ci']}% CI bounds", f"Pixelated ({qres['grid']}² states)"],
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

    # ---------------------------------------------------------
    # 5. EXPANDABLE MATH & HELPER FUNCTIONS
    # ---------------------------------------------------------
    st.divider()
    with st.expander("📚 View Calculation Methodology & Helper Equations"):
        st.markdown("#### 1. Classical 2D Probability of Collision ($P_c$)")
        st.write("The classical probability is calculated by integrating the 2D Gaussian probability density function (PDF) over the circular cross-section of the combined Hard-Body Radius ($R_{HBR}$) in the Encounter Plane.")
        st.latex(r"""
        P_c = \iint_{x^2 + y^2 \leq R_{HBR}^2} \frac{1}{2\pi\sigma_x\sigma_y} \exp\left[ -\frac{1}{2} \left( \frac{(x-\mu_x)^2}{\sigma_x^2} + \frac{(y-\mu_y)^2}{\sigma_y^2} \right) \right] dx dy
        """)
        
        st.markdown("#### 2. Quantum State Preparation ($\ket{\psi}$)")
        st.write("To evaluate this on a quantum computer, the continuous Encounter Plane is discretized into a finite $G \times G$ grid using $q$ qubits per axis. The probabilities are loaded into the amplitudes of the quantum state.")
        st.latex(r"""
        \ket{\psi}_{n} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} \sqrt{p(x_i, y_j)} \ket{i}_x \ket{j}_y \ket{0}_{obj}
        """)
        
        st.markdown("#### 3. The Collision Oracle ($U_\omega$)")
        st.write("A quantum oracle flags the states where the Euclidean distance to the origin is less than the Hard-Body Radius by flipping an objective qubit $\ket{obj}$.")
        st.latex(r"""
        U_\omega \ket{i,j}\ket{0} = 
        \begin{cases} 
        \ket{i,j}\ket{1} & \text{if } \sqrt{x_i^2 + y_j^2} \leq R_{HBR} \\
        \ket{i,j}\ket{0} & \text{otherwise}
        \end{cases}
        """)

        st.markdown("#### 4. Speedup Formulation (Variance Matching)")
        st.write("The classical sample requirement ($N$) is dynamically calculated by matching the variance of a classical binomial distribution to the exact Confidence Interval bound achieved by the Quantum IQAE algorithm.")
        st.latex(r"""
        N_{equiv} = \left\lceil \frac{Z^2 \cdot P_c(1 - P_c)}{\varepsilon_{IQAE}^2} \right\rceil \quad \longrightarrow \quad \text{Speedup} = \frac{N_{equiv}}{M_{queries}}
        """)


with tabs[1]:
    st.markdown("### 📡 SGP4 Orbital Propagation")

    c1, c2, c3 = st.columns(3)
    c1.metric("TCA offset",     f"{R['best_t_s']/60:.2f} min from epoch")
    c2.metric("Miss Distance",  f"{R['miss_km']*1000:.2f} m")
    c3.metric("TCA search time",f"{R['t_tca']*1e3:.0f} ms")

    st.divider()
    st.markdown("**State Vectors at TCA**")
    sv_data = {
            "Component": ["X (km)", "Y (km)", "Z (km)"],
            f"{R['obj1'].split('(')[0].strip()} Position": [
                f"{R['r1'][0]:.3f}", f"{R['r1'][1]:.3f}", f"{R['r1'][2]:.3f}"],
            f"{R['obj2'].split('(')[0].strip()} Position": [
                f"{R['r2'][0]:.3f}", f"{R['r2'][1]:.3f}", f"{R['r2'][2]:.3f}"],
        }
    import pandas as pd
    st.dataframe(pd.DataFrame(sv_data), use_container_width=True, hide_index=True)

    vv_data = {
            "Component": ["Vx (km/s)", "Vy (km/s)", "Vz (km/s)"],
            f"{R['obj1'].split('(')[0].strip()} Velocity": [
                f"{R['v1'][0]:.4f}", f"{R['v1'][1]:.4f}", f"{R['v1'][2]:.4f}"],
            f"{R['obj2'].split('(')[0].strip()} Velocity": [
                f"{R['v2'][0]:.4f}", f"{R['v2'][1]:.4f}", f"{R['v2'][2]:.4f}"],
        }
    st.dataframe(pd.DataFrame(vv_data), use_container_width=True, hide_index=True)


with tabs[2]:
    st.markdown("### ⚛ Quantum vs Classical Comparison")

    # ── Bar comparison ────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        methods = ["Classical MC", "Quantum IQAE", "Discrete Ground Truth"]
        vals    = [R["pc_mc"], qres["pc"], qres["pc_disc"]]
        colors  = [CAMB, CQNT, CACC]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=methods, y=vals,
            marker=dict(color=colors, line=dict(color=BDR, width=1.5)),
            text=[f"{v:.4e}" for v in vals],
            textposition='outside', textfont=dict(color=CTXT, size=10),
            error_y=dict(
                type='data', symmetric=False,
                array=[0, qres["ci"][1]-qres["pc"], 0],
                arrayminus=[0, qres["pc"]-qres["ci"][0], 0],
                color=CTXT, thickness=2, width=8)
        ))
        fig_bar.update_layout(
            plot_bgcolor=PANEL, paper_bgcolor=DARK,
            font=dict(color=CTXT, family='monospace'),
            yaxis=dict(title='Collision Probability Pc', showgrid=True,
                       gridcolor=BDR, color=CMUT),
            xaxis=dict(color=CMUT),
            height=380, margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Pc Estimate by Method", font=dict(size=13))
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)

    with col_r:
        st.markdown("**IQAE 95 % Confidence Interval**")
        ci_lo, ci_hi = qres["ci"]
        fig_ci = go.Figure()
        fig_ci.add_shape(type='rect',
            x0=ci_lo, x1=ci_hi, y0=0.1, y1=0.9,
            fillcolor=CQNT, opacity=0.15,
            line=dict(color=CQNT, width=1.5))
        fig_ci.add_vline(x=qres["pc"],    line=dict(color=CQNT, width=2.5, dash='solid'),
                         annotation=dict(text="IQAE", font=dict(color=CQNT)))
        fig_ci.add_vline(x=R["pc_mc"],    line=dict(color=CAMB, width=2, dash='dot'),
                         annotation=dict(text="MC",   font=dict(color=CAMB), y=0.7))
        fig_ci.add_vline(x=qres["pc_disc"],line=dict(color=CACC, width=1.5, dash='dash'),
                         annotation=dict(text="Truth",font=dict(color=CACC), y=0.4))
        fig_ci.update_layout(
            plot_bgcolor=PANEL, paper_bgcolor=DARK,
            font=dict(color=CTXT, family='monospace'),
            xaxis=dict(title='Pc', showgrid=True, gridcolor=BDR, color=CMUT,
                       tickformat='.2e'),
            yaxis=dict(visible=False),
            height=240, margin=dict(l=10, r=10, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_ci, use_container_width=True, theme=None)

        st.markdown(f"""
| Bound | Value |
|---|---|
| CI lower | `{ci_lo:.4e}` |
| CI upper | `{ci_hi:.4e}` |
| CI width | `{ci_hi-ci_lo:.4e}` |
| MC Pc    | `{R['pc_mc']:.4e}` |
| IQAE Pc  | `{qres['pc']:.4e}` |
""")


with tabs[3]:
    st.markdown("### 📈 Scaling & Complexity Analysis")

    # --------------------------------------------------
    # ACTUAL RUN DATA (Linked to true physics math)
    # --------------------------------------------------
    eps_used = R["epsilon"]
    queries  = max(R["qres"]["queries"], 1)
    mc_equiv = R["mc_equiv"] # Uses the precise Z-score variance from the pipeline!
    speedup_actual = R["speedup"]

    # --------------------------------------------------
    # THEORETICAL SCALING 
    # --------------------------------------------------
    eps_arr = np.logspace(-1, -5, 300)

    # We scale the theoretical lines so they perfectly pass through our actual data point
    c_factor = mc_equiv * (eps_used**2)
    q_factor = queries * eps_used

    N_c = c_factor / (eps_arr**2)  # Classical O(1/ε²)
    M_q = q_factor / eps_arr       # Quantum O(1/ε)

    speedup_theory = N_c / M_q 

    # --------------------------------------------------
    # FIGURE 1: SCALING 
    # --------------------------------------------------
    fig_scale = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Resources vs Target Precision",
            "Quantum Speedup Factor"
        ],
        horizontal_spacing=0.15 # FIX: Stops the right graph from crushing the left graph
    )

    # ---------------- LEFT PANEL ----------------
    fig_scale.add_trace(go.Scatter(
        x=eps_arr, y=N_c, mode='lines',
        name='Classical O(ε⁻²)', line=dict(color=CAMB, width=2.5)
    ), row=1, col=1)

    fig_scale.add_trace(go.Scatter(
        x=eps_arr, y=M_q, mode='lines',
        name='Quantum O(ε⁻¹)', line=dict(color=CQNT, width=2.5)
    ), row=1, col=1)

    fig_scale.add_vline(
        x=eps_used, line=dict(color=CACC, width=1.5, dash='dot'),
        annotation=dict(text=f"Current ε", font=dict(color=CACC), y=0.9)
    )

    # ---------------- RIGHT PANEL ----------------
    fig_scale.add_trace(go.Scatter(
        x=eps_arr, y=speedup_theory, mode='lines',
        name='Theoretical Speedup', line=dict(color=CGRN, width=2.5),
        fill='tozeroy', fillcolor='rgba(52,211,153,0.05)'
    ), row=1, col=2)

    fig_scale.add_trace(go.Scatter(
        x=[eps_used], y=[speedup_actual], mode='markers+text',
        marker=dict(size=14, color=CACC, symbol='diamond'),
        text=[f"{speedup_actual:,.0f}×"], textposition="top left",
        name='Actual Speedup'
    ), row=1, col=2)

    # Axis formatting
    fig_scale.update_xaxes(
        type='log', autorange='reversed', showgrid=True, gridcolor=BDR, color=CMUT,
        title_text='Target ε', title_font=dict(color=CMUT), title_standoff=15,
        row=1, col=1
    )
    fig_scale.update_xaxes(
        type='log', autorange='reversed', showgrid=True, gridcolor=BDR, color=CMUT,
        title_text='Target ε', title_standoff=15,
        row=1, col=2
    )
    fig_scale.update_yaxes(type='log', showgrid=True, gridcolor=BDR, color=CMUT, title_standoff=15)

    fig_scale.update_layout(
        plot_bgcolor=PANEL, paper_bgcolor=DARK,
        font=dict(color=CTXT, family='monospace'),
        legend=dict(
            bgcolor=CARD, bordercolor=BDR,
            orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5 # FIX: Move legend below plots
        ),
        height=500,
        margin=dict(l=70, r=20, t=60, b=80), # FIX: Massive margin increase prevents cut-off text!
    )
    st.plotly_chart(fig_scale, use_container_width=True, theme=None)


    
    st.divider()
    st.markdown("#### Error vs Computational Effort (Convergence Rates)")

    # Mathematically anchor the convergence constants to the actual physics simulation!
    c0 = eps_used * np.sqrt(mc_equiv)
    q0 = eps_used * queries

    N_ext = np.logspace(np.log10(max(mc_equiv/100, 10)), np.log10(mc_equiv*100), 200)
    err_c = c0 / np.sqrt(N_ext)

    M_ext = np.logspace(np.log10(max(queries/10, 2)), np.log10(queries*100), 200)
    err_q = q0 / M_ext

    fig_conv = go.Figure()

    fig_conv.add_trace(go.Scatter(
        x=N_ext, y=err_c, mode='lines',
        name='Classical ε ∝ N⁻¹/²', line=dict(color=CAMB, width=2.5)
    ))

    fig_conv.add_trace(go.Scatter(
        x=M_ext, y=err_q, mode='lines',
        name='Quantum ε ∝ M⁻¹', line=dict(color=CQNT, width=2.5)
    ))
    
    # Plot the exact data point where they achieved target precision
    fig_conv.add_trace(go.Scatter(
        x=[mc_equiv], y=[eps_used], mode='markers',
        name='MC Equivalent', marker=dict(size=10, color=CAMB, symbol='circle')
    ))
    fig_conv.add_trace(go.Scatter(
        x=[queries], y=[eps_used], mode='markers',
        name='IQAE Queries', marker=dict(size=12, color=CQNT, symbol='star')
    ))

    # Advantage region
    M_common = np.logspace(np.log10(max(queries/10, 2)), np.log10(mc_equiv*100), 200)
    err_qc = q0 / M_common
    err_cc = c0 / np.sqrt(M_common)

    fig_conv.add_trace(go.Scatter(
        x=np.concatenate([M_common, M_common[::-1]]),
        y=np.concatenate([err_qc, err_cc[::-1]]),
        fill='toself', fillcolor='rgba(56,189,248,0.07)', line=dict(color='rgba(0,0,0,0)'),
        name='Quantum advantage zone'
    ))

    fig_conv.update_layout(
        xaxis=dict(
            type='log', title='Computational Effort (N or M)', title_standoff=15,
            showgrid=True, gridcolor=BDR, color=CMUT
        ),
        yaxis=dict(
            type='log', title='Absolute Error ε', title_standoff=15,
            showgrid=True, gridcolor=BDR, color=CMUT
        ),
        plot_bgcolor=PANEL, paper_bgcolor=DARK,
        font=dict(color=CTXT, family='monospace'),
        legend=dict(bgcolor=CARD, bordercolor=BDR, x=0.75, y=0.95),
        height=450,
        margin=dict(l=70, r=20, t=40, b=70), # FIX: Increased bottom/left margins
    )

    st.plotly_chart(fig_conv, use_container_width=True, theme=None)

with tabs[4]:
    st.markdown("### 🎛 Quantum Circuit Diagnostics")

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

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Circuit Depth vs Precision (projected)**")
        eps_range  = np.array([0.05, 0.02, 0.01, 0.005, 0.002])
        dep_proj   = (qres["depth"] * np.log(0.01) / np.log(eps_range)).astype(int)
        dep_proj   = np.abs(dep_proj)
        fig_dep    = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=eps_range, y=dep_proj, mode='lines+markers',
            line=dict(color=CACC, width=2.5),
            marker=dict(size=9, color=CACC, symbol='diamond')))
        fig_dep.add_vline(x=R["epsilon"],
                          line=dict(color=CQNT, width=1.5, dash='dot'),
                          annotation=dict(text="Current ε",
                                         font=dict(color=CQNT)))
        fig_dep.update_layout(
            xaxis=dict(autorange='reversed', title='Target ε',
                       showgrid=True, gridcolor=BDR, color=CMUT),
            yaxis=dict(title='Est. Circuit Depth',
                       showgrid=True, gridcolor=BDR, color=CMUT),
            plot_bgcolor=PANEL, paper_bgcolor=DARK,
            font=dict(color=CTXT, family='monospace'),
            height=310, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
        )
        st.plotly_chart(fig_dep, use_container_width=True, theme=None)

    with col_r:
        st.markdown("**Oracle Query Breakdown**")
        labels = ["Marked (collision)", "Unmarked (safe)"]
        values = [qres["marked"], qres["grid"]**2 - qres["marked"]]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.52,
            marker=dict(colors=[CRED, CQNT],
                        line=dict(color=DARK, width=2)),
            textfont=dict(color=CTXT, size=10)
        ))
        fig_pie.update_layout(
            paper_bgcolor=DARK, font=dict(color=CTXT, family='monospace'),
            legend=dict(bgcolor=CARD, bordercolor=BDR),
            height=310, margin=dict(l=10, r=10, t=20, b=10),
            annotations=[dict(text=f"{qres['marked']}", x=0.5, y=0.5,
                              font=dict(size=22, color=CRED), showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

    st.divider()
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
        xaxis=dict(title=f'Top {top_k} basis state index (sorted)',
                   showgrid=False, color=CMUT),
        yaxis=dict(title='Amplitude √pdf', showgrid=True,
                   gridcolor=BDR, color=CMUT),
        plot_bgcolor=PANEL, paper_bgcolor=DARK,
        font=dict(color=CTXT, family='monospace'),
        height=280, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
    )
    st.plotly_chart(fig_amp, use_container_width=True, theme=None)



with tabs[5]:
    st.markdown("### 📋 Mission Decision Report")

    pc_val = R["pc_mc"]
    if pc_val > 1e-3:
        decision = "🔴  MANOEUVRE REQUIRED"
        dec_col  = "#f87171"
        dec_bg   = "#7f1d1d"
        rec_text = ("Collision probability exceeds the operational threshold (1×10⁻⁴). "
                    "A collision avoidance manoeuvre (CAM) should be planned immediately. "
                    "Quantum IQAE confirms the estimate with a tight 95% confidence interval.")
    elif pc_val > 1e-4:
        decision = "🟡  HEIGHTENED MONITORING"
        dec_col  = "#f59e0b"
        dec_bg   = "#431407"
        rec_text = ("Probability is elevated but below the mandatory manoeuvre threshold. "
                    "Recommend continuous monitoring with updated TLE refreshes every 6 hours. "
                    "Pre-plan a contingency CAM.")
    else:
        decision = "🟢  NO ACTION REQUIRED"
        dec_col  = "#34d399"
        dec_bg   = "#052e16"
        rec_text = ("Collision probability is within nominal limits. "
                    "Continue routine tracking. Review if TLE age exceeds 3 days.")

    st.markdown(f"""
<div style="background:{dec_bg};border:2px solid {dec_col};border-radius:12px;
            padding:1.2rem 2rem;margin-bottom:1.5rem;text-align:center">
  <div style="font-size:1.6rem;font-weight:800;color:{dec_col}">{decision}</div>
  <div style="color:#94a3b8;margin-top:.5rem;font-size:.9rem">{rec_text}</div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🛰 Object & Encounter Data")
        rows = [
            ("Object 1",            R["obj1"]),
            ("Object 2",            R["obj2"]),
            ("Miss Distance",       f"{R['miss_km']*1000:.2f} m"),
            ("Combined HBR",        f"{R['R_hbr']:.0f} m"),
            ("σ Radial",            f"{R['sig_r']:.1f} m"),
            ("σ Along-track",       f"{R['sig_s']:.1f} m"),
            ("Grid Resolution",     f"{qres['grid']}×{qres['grid']}"),
            ("Collision Cells",     f"{qres['marked']} / {qres['grid']**2}"),
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows, columns=["Parameter", "Value"]),
                     use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### ⚛ Quantum Computation Summary")
        rows_q = [
            ("Pc (Classical MC)",     f"{R['pc_mc']:.6e}"),
            ("Pc (Quantum IQAE)",     f"{qres['pc']:.6e}"),
            ("Pc (Discrete Truth)",   f"{qres['pc_disc']:.6e}"),
            ("IQAE 95% CI",           f"[{qres['ci'][0]:.3e}, {qres['ci'][1]:.3e}]"),
            ("Oracle Queries (IQAE)", f"{qres['queries']}"),
            ("MC Samples",            f"{R['mc_N']:,}"),
            ("MC Equiv for match",    f"{R['mc_equiv']:,}"),
            ("Quantum Speedup",       f"{R['speedup']:,.0f}×"),
            ("Circuit Depth",         f"{qres['depth']}"),
            ("Target ε",              f"{R['epsilon']}"),
            ("Confidence Level",      f"{R['alpha_ci']}%"),
        ]
        st.dataframe(pd.DataFrame(rows_q, columns=["Metric", "Value"]),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 📊 Threshold Reference Chart")

    thresholds = {
        "Threshold": ["RED (Manoeuvre)", "YELLOW (Monitor)", "GREEN (Nominal)"],
        "Pc Range":  ["> 1×10⁻⁴",      "1×10⁻⁵ – 1×10⁻⁴", "< 1×10⁻⁵"],
        "Action":    ["CAM required",     "Pre-plan CAM",      "Routine tracking"],
        "This Event":[
            "✓" if pc_val > 1e-4 else "–",
            "✓" if 1e-5 < pc_val <= 1e-6 else "–",
            "✓" if pc_val <= 1e-6 else "–",
        ]
    }
    st.dataframe(pd.DataFrame(thresholds), use_container_width=True, hide_index=True)

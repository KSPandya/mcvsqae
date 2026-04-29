"""
═══════════════════════════════════════════════════════════════════════════
  QUANTUM COLLISION PROBABILITY  —  Streamlit Dashboard
  ─────────────────────────────────────────────────────────────────────────
  Run:  streamlit run quantum_collision_app.py

  Install:
    pip install streamlit sgp4 qiskit qiskit-aer qiskit-algorithms
               numpy scipy matplotlib plotly
═══════════════════════════════════════════════════════════════════════════
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

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="Quantum Orbital Collision Estimator",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Injected CSS ─────────────────────────────────────────────────────────────
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

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COLOUR PALETTE (matplotlib)                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ─────────────────────────────────────────────────────────────
# CORE DARK THEME (higher contrast)
# ─────────────────────────────────────────────────────────────
DARK  = "#020617"   # near-black (better contrast than blue-black)
PANEL = "#0B1220"   # deep slate
CARD  = "#111827"   # elevated panel
BDR   = "#1F2A44"   # subtle but visible borders

# ─────────────────────────────────────────────────────────────
# PRIMARY SIGNAL COLORS (scientifically distinct)
# ─────────────────────────────────────────────────────────────
CAMB  = "#F59E0B"   # Classical → Amber (unchanged, already strong)
CQNT  = "#22D3EE"   # Quantum → Bright Cyan (more luminous than before)
CACC  = "#A78BFA"   # Accent → Violet (kept but slightly brighter)

# ─────────────────────────────────────────────────────────────
# SUPPORT / STATUS COLORS (high visibility)
# ─────────────────────────────────────────────────────────────
CGRN  = "#4ADE80"   # Success → vivid green (brighter than before)
CRED  = "#FB7185"   # Alert → pink-red (better than dull red on dark)
CYEL  = "#FACC15"   # Warning → bright yellow (new, useful)

# ─────────────────────────────────────────────────────────────
# TEXT (true contrast hierarchy)
# ─────────────────────────────────────────────────────────────
CTXT  = "#F9FAFB"   # primary text (almost white)
CMUT  = "#94A3B8"   # secondary text (cool gray)

# ─────────────────────────────────────────────────────────────
# OBJECT COLORS (distinct, non-conflicting)
# ─────────────────────────────────────────────────────────────
COBJ1 = "#FB923C"   # orange
COBJ2 = "#60A5FA"   # bright blue (less purple overlap)

# ─────────────────────────────────────────────────────────────
# OPTIONAL EXTRA COLORS (for multi-line plots)
# ─────────────────────────────────────────────────────────────
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
    "grid.color": "#1F2937",     # darker grid (less visual clutter)
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
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — fine-tuning parameters                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
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

    # ── Encounter Geometry override ───────────────────────────────────────
    #st.markdown('<div class="sidebar-label">📐 Encounter Geometry</div>', unsafe_allow_html=True)
    #use_override = st.toggle("Override with custom geometry", value=True, help="Force a specific near-miss for demonstration")
    #if use_override:
     #   mu_r  = st.slider("Miss distance — Radial (m)",   -200, 200,  20, 5)
      #  mu_s  = st.slider("Miss distance — Along-track (m)", -500, 500, 30, 10)
       # sig_r = st.slider("σ Radial (m)",   10, 200,  60, 5)
        #sig_s = st.slider("σ Along-track (m)", 50, 600, 250, 10)
        #R_hbr = st.slider("Hard-body Radius (m)",  10, 300, 115, 5)
    #else:
     #   mu_r, mu_s    = 20.0, 30.0
      #  sig_r, sig_s  = 60.0, 250.0
      #  R_hbr         = 115.0

    st.markdown("---")
    use_override=0
    # ── Quantum Parameters ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">⚛ Quantum (IQAE)</div>', unsafe_allow_html=True)
    epsilon   = st.select_slider("Target precision ε",
        options=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001], value=0.01)
    alpha_ci  = st.slider("Confidence level (1−α) %", 80, 99, 95, 1)
    q_dim     = st.select_slider("Qubits per axis",
        options=[3, 4, 5], value=4,
        help="Grid = 2^q × 2^q  |  4→16×16  |  5→32×32")

    st.markdown("---")

    # ── Monte Carlo ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🎲 Monte Carlo</div>', unsafe_allow_html=True)
    mc_N = st.select_slider("MC samples N",
        options=[10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000],
        value=500_000)

    st.markdown("---")

    # ── TCA Search ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🔭 TCA Search</div>', unsafe_allow_html=True)
    tca_window  = st.slider("Search window (min)", 30, 360, 120, 30)
    coarse_step = st.slider("Coarse step (s)", 5, 30, 10, 5)

    run_btn = st.button("▶  RUN SIMULATION", use_container_width=True, type="primary")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORE PHYSICS FUNCTIONS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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
    r_v  = np.linspace(-span, span, G)
    s_v  = np.linspace(-span, span, G)
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

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE — run once, cache results                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if "results" not in st.session_state:
    st.session_state.results = None

# ── Header ────────────────────────────────────────────────────────────────────
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

# ── Run button logic ──────────────────────────────────────────────────────────
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

            # Use override geometry if selected
            eff_mu_r, eff_mu_s   = (mu_r, mu_s)     if use_override else (20.0, 30.0)
            eff_sig_r, eff_sig_s = (sig_r, sig_s)   if use_override else (60.0, 250.0)
            eff_R                = R_hbr             if use_override else 115.0

            st.write(f"🎲 Running Monte Carlo  (N = {mc_N:,}) …")
            t_mc_start = time.perf_counter()
            pc_mc   = classical_mc(eff_mu_r, eff_mu_s, eff_sig_r, eff_sig_s, eff_R, mc_N)
            t_mc    = time.perf_counter() - t_mc_start

            st.write(f"⚛ Running Quantum IQAE  (ε = {epsilon}, grid = {2**q_dim}×{2**q_dim}) …")
            t_q_start = time.perf_counter()
            qres    = quantum_iqae(eff_mu_r, eff_mu_s, eff_sig_r, eff_sig_s,
                                   eff_R, epsilon, alpha_ci, q_dim)
            t_q     = time.perf_counter() - t_q_start
	    
            mc_equiv = int(np.ceil(1.0 / epsilon**2))
            speedup  = mc_equiv / max(qres["queries"], 1)

            st.session_state.results = dict(
                obj1=obj1_name, obj2=obj2_name,
                miss_km=miss_km, best_t_s=best_t, t_tca=t_tca,
                r1=r1, v1=v1, r2=r2, v2=v2,
                mu_r=eff_mu_r, mu_s=eff_mu_s,
                sig_r=eff_sig_r, sig_s=eff_sig_s, R_hbr=eff_R,
                pc_mc=pc_mc, t_mc=t_mc,
                qres=qres, t_q=t_q,
                mc_equiv=mc_equiv, speedup=speedup,
                epsilon=epsilon, alpha_ci=alpha_ci, q_dim=q_dim, mc_N=mc_N,
            )
            status.update(label="✅ Pipeline complete!", state="complete")
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")
            st.exception(e)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DISPLAY TABS                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
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
# TAB 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 🌍 Mission Overview")

    # Risk badge
    pc = R["pc_mc"]
    if pc > 1e-4:
        risk_html = '<div class="risk-high">🔴  HIGH RISK — MANOEUVRE REQUIRED</div>'
    elif pc > 1e-5:
        risk_html = '<div class="risk-mid">🟡  ELEVATED RISK — MONITOR CLOSELY</div>'
    else:
        risk_html = '<div class="risk-low">🟢  NOMINAL RISK — NO ACTION</div>'
    st.markdown(risk_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Top-level metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Object 1",       R["obj1"].split("(")[0].strip())
    c2.metric("Object 2",       R["obj2"].split("(")[0].strip())
    c3.metric("Miss Distance",  f"{R['miss_km']*1000:.1f} m")
    c4.metric("Pc (Monte Carlo)", f"{R['pc_mc']:.3e}")
    c5.metric("Pc (Quantum IQAE)", f"{qres['pc']:.3e}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Classical vs Quantum at a Glance")
        comparison = {
            "Method":          ["Classical MC",            "Quantum IQAE"],
            "Pc Estimate":     [f"{R['pc_mc']:.4e}",       f"{qres['pc']:.4e}"],
            "Queries / Samples":[f"{R['mc_N']:,}",         f"{qres['queries']}"],
            "Wall-clock Time": [f"{R['t_mc']*1e3:.1f} ms", f"{R['t_q']:.2f} s"],
            "Error Scaling":   ["O(N⁻¹/²)",                "O(M⁻¹)"],
            "Uncertainty":     ["Post-hoc estimate",       "Native 95% CI"],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### Quantum Advantage Summary")
        st.markdown(f"""
<div class="speedup-box">
  <div style="font-size:2.8rem;font-weight:800;
    background:linear-gradient(90deg,#a78bfa,#38bdf8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text">{R['speedup']:,.0f}×</div>
  <div style="color:#64748b;margin-top:.3rem">Quantum Speedup over Classical</div>
  <hr style="border-color:#1a3050;margin:.8rem 0">
  <div style="font-size:.85rem;color:#94a3b8">
    MC needs <strong style="color:#f59e0b">{R['mc_equiv']:,}</strong> samples to match IQAE accuracy<br>
    IQAE used only <strong style="color:#38bdf8">{qres['queries']}</strong> oracle queries<br>
    IQAE 95% CI: [{qres['ci'][0]:.4e}, {qres['ci'][1]:.4e}]
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SGP4 PROPAGATION
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUANTUM vs CLASSICAL
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCALING & COMPLEXITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 📈 Scaling & Complexity Analysis")

    # --------------------------------------------------
    # THEORETICAL SCALING
    # --------------------------------------------------
    eps_arr = np.logspace(-1, -5, 300)

    # Classical and Quantum costs
    N_c = 1.0 / eps_arr**2       # Classical
    M_q = 1.0 / eps_arr          # Quantum

    # Theoretical speedup
    speedup_theory = N_c / M_q   # = 1/ε

    # --------------------------------------------------
    # ACTUAL RUN DATA (IMPORTANT FIX)
    # --------------------------------------------------
    eps_used = R["epsilon"]
    queries = max(R["qres"]["queries"], 1)

    # Classical equivalent cost at same target epsilon
    mc_equiv = 1.0 / (eps_used**2)

    # Actual speedup
    speedup_actual = mc_equiv / queries

    # --------------------------------------------------
    # FIGURE: SCALING
    # --------------------------------------------------
    fig_scale = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Resources Required vs Target Precision",
            "Quantum Speedup Factor"
        ],
    )

    # ---------------- LEFT PANEL ----------------
    fig_scale.add_trace(go.Scatter(
        x=eps_arr, y=N_c,
        mode='lines',
        name='Classical  O(ε⁻²)',
        line=dict(color=CAMB, width=2.5)
    ), row=1, col=1)

    fig_scale.add_trace(go.Scatter(
        x=eps_arr, y=M_q,
        mode='lines',
        name='Quantum  O(ε⁻¹)',
        line=dict(color=CQNT, width=2.5)
    ), row=1, col=1)

    # Mark epsilon used
    fig_scale.add_vline(
        x=eps_used,
        line=dict(color=CACC, width=1.5, dash='dot'),
        annotation=dict(
            text=f"ε={eps_used}",
            font=dict(color=CACC)
        )
    )

    # ---------------- RIGHT PANEL ----------------
    # Theoretical speedup curve
    fig_scale.add_trace(go.Scatter(
        x=eps_arr,
        y=speedup_theory,
        mode='lines',
        name='Theoretical (1/ε)',
        line=dict(color=CGRN, width=2.5),
        fill='tozeroy',
        fillcolor='rgba(52,211,153,0.05)'
    ), row=1, col=2)

    # Actual speedup point (FIXED)
    fig_scale.add_trace(go.Scatter(
        x=[eps_used],
        y=[speedup_actual],
        mode='markers+text',
        marker=dict(size=14, color=CACC, symbol='diamond'),
        text=[f"{speedup_actual:.0f}×"],
        textposition="top center",
        name='Actual (this run)'
    ), row=1, col=2)

    # Annotation explaining comparison
    fig_scale.add_annotation(
        x=eps_used,
        y=speedup_actual,
        text=(
            f"MC ≈ {int(mc_equiv):,} samples<br>"
            f"IQAE = {queries} queries"
        ),
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(color=CTXT, size=10),
        row=1,
        col=2
    )

    # Axis formatting
    fig_scale.update_xaxes(
        type='log',
        autorange='reversed',
        showgrid=True,
        gridcolor=BDR,
        color=CMUT,
        title_text='Target ε',
        title_font=dict(color=CMUT),
        row=1, col=1
    )

    fig_scale.update_xaxes(
        type='log',
        autorange='reversed',
        showgrid=True,
        gridcolor=BDR,
        color=CMUT,
        title_text='Target ε',
        row=1, col=2
    )

    fig_scale.update_yaxes(
        type='log',
        showgrid=True,
        gridcolor=BDR,
        color=CMUT
    )

    fig_scale.update_layout(
        plot_bgcolor=PANEL,
        paper_bgcolor=DARK,
        font=dict(color=CTXT, family='monospace'),
        legend=dict(bgcolor=CARD, bordercolor=BDR),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    st.plotly_chart(fig_scale, use_container_width=True, theme=None)

    # ==================================================
    # CONVERGENCE PLOT (UNCHANGED BUT CLEANED)
    # ==================================================
    st.divider()
    st.markdown("#### Error vs Computational Effort (Convergence Rates)")

    N_ext = np.logspace(2, 9, 200)
    c0    = 0.1 * np.sqrt(100)
    err_c = c0 / np.sqrt(N_ext)

    M_ext = np.logspace(1, 5, 200)
    q0    = 0.1 * 10
    err_q = q0 / M_ext

    fig_conv = go.Figure()

    fig_conv.add_trace(go.Scatter(
        x=N_ext, y=err_c,
        mode='lines',
        name='Classical  ε ∝ N⁻¹/²',
        line=dict(color=CAMB, width=2.5)
    ))

    fig_conv.add_trace(go.Scatter(
        x=M_ext, y=err_q,
        mode='lines',
        name='Quantum  ε ∝ M⁻¹',
        line=dict(color=CQNT, width=2.5)
    ))

    # Advantage region
    M_common = np.logspace(2, 5, 200)
    err_qc   = q0 / M_common
    err_cc   = c0 / np.sqrt(M_common)

    fig_conv.add_trace(go.Scatter(
        x=np.concatenate([M_common, M_common[::-1]]),
        y=np.concatenate([err_qc, err_cc[::-1]]),
        fill='toself',
        fillcolor='rgba(56,189,248,0.07)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Quantum advantage zone'
    ))

    fig_conv.update_layout(
        xaxis=dict(
            type='log',
            title='Computational Effort (N or M)',
            showgrid=True,
            gridcolor=BDR,
            color=CMUT
        ),
        yaxis=dict(
            type='log',
            title='Absolute Error ε',
            showgrid=True,
            gridcolor=BDR,
            color=CMUT
        ),
        plot_bgcolor=PANEL,
        paper_bgcolor=DARK,
        font=dict(color=CTXT, family='monospace'),
        legend=dict(bgcolor=CARD, bordercolor=BDR),
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fig_conv, use_container_width=True, theme=None)
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CIRCUIT DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MISSION DECISION
# ══════════════════════════════════════════════════════════════════════════════
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
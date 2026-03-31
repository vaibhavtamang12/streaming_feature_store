"""
Streamlit real-time fraud dashboard.
Polls the inference API and Prometheus for live metrics.
"""
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Fraud Detection Console",
    page_icon="🛡️",
    layout="wide",
)

# ── Glassmorphism CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
  }
  .metric-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 20px 24px;
    color: white;
  }
  .metric-value { font-size: 2.2rem; font-weight: 600; }
  .metric-label { font-size: 0.85rem; opacity: 0.7; margin-bottom: 4px; }
  .fraud-badge {
    background: rgba(220, 53, 69, 0.25);
    border: 1px solid rgba(220, 53, 69, 0.5);
    border-radius: 8px; padding: 2px 10px;
    color: #ff6b7a; font-size: 0.8rem;
  }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_volume" not in st.session_state:
    st.session_state.total_volume = 0
if "total_blocked" not in st.session_state:
    st.session_state.total_blocked = 0

API_URL = "http://localhost:8000"
PROM_URL = "http://localhost:9090"

def query_prometheus(promql: str) -> float:
    try:
        r = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": promql},
            timeout=2,
        )
        result = r.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return 0.0

def get_live_metrics() -> dict:
    return {
        "p95_latency_ms": query_prometheus(
            'histogram_quantile(0.95, fraud_api_request_duration_seconds_bucket)') * 1000,
        "rps": query_prometheus('rate(fraud_detections_total[1m]) + rate(legit_transactions_total[1m])'),
        "fraud_rate": query_prometheus(
            'rate(fraud_detections_total[5m]) / (rate(fraud_detections_total[5m]) + rate(legit_transactions_total[5m]))'),
    }

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown("## 🛡️ Fraud Detection Console")
st.markdown("---")

placeholder = st.empty()
refresh_rate = st.sidebar.slider("Refresh rate (sec)", 1, 10, 2)
confidence_threshold = st.sidebar.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.05)

while True:
    metrics = get_live_metrics()

    with placeholder.container():
        # ── KPI row ──────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""
          <div class="metric-card">
            <div class="metric-label">Total Volume</div>
            <div class="metric-value">{st.session_state.total_volume:,}</div>
          </div>""", unsafe_allow_html=True)
        c2.markdown(f"""
          <div class="metric-card">
            <div class="metric-label">Blocked Transactions</div>
            <div class="metric-value" style="color:#ff6b7a">{st.session_state.total_blocked:,}</div>
          </div>""", unsafe_allow_html=True)
        c3.markdown(f"""
          <div class="metric-card">
            <div class="metric-label">P95 Latency</div>
            <div class="metric-value">{metrics['p95_latency_ms']:.1f} ms</div>
          </div>""", unsafe_allow_html=True)
        c4.markdown(f"""
          <div class="metric-card">
            <div class="metric-label">Fraud Rate (5m)</div>
            <div class="metric-value">{metrics['fraud_rate']*100:.2f}%</div>
          </div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Live confidence score stream ───────────────────────────────────
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            if st.session_state.history:
                df = pd.DataFrame(st.session_state.history[-60:])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["fraud_probability"],
                    mode="lines+markers",
                    line=dict(color="rgba(255,107,122,0.9)", width=2),
                    marker=dict(
                        color=df["is_fraud"].map({True: "#ff6b7a", False: "#4ecdc4"}),
                        size=6,
                    ),
                    name="Fraud probability",
                ))
                fig.add_hline(
                    y=confidence_threshold, line_dash="dash",
                    line_color="rgba(255,255,255,0.4)",
                    annotation_text="Threshold",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.04)",
                    font_color="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                    margin=dict(l=0, r=0, t=30, b=0),
                    title="Model confidence — live stream",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**Recent detections**")
            if st.session_state.history:
                recent = pd.DataFrame(st.session_state.history[-10:][::-1])
                recent["status"] = recent["is_fraud"].map({True: "🔴 FRAUD", False: "🟢 LEGIT"})
                st.dataframe(
                    recent[["timestamp", "fraud_probability", "status"]],
                    hide_index=True,
                    use_container_width=True,
                )

    time.sleep(refresh_rate)
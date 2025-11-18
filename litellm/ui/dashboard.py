import streamlit as st
import streamlit_shadcn as shadcn
from streamlit_shadcn import card, button, metric

st.set_page_config(page_title="Skypiea Gateway", layout="wide", initial_sidebar_state="expanded")

# Hero Section
st.markdown("""
<div style='text-align: center;'>
  <h1 style='color: #0ea5e9;'>🚀 Skypiea Gateway</h1>
  <p>Modern LLM Proxy | Vision + Reasoning + Tools | 250+ Providers</p>
</div>
""", unsafe_allow_html=True)

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1: metric("Total Calls", "12.3k", delta="↑ 23%")
with col2: metric("Cost Saved", "$45.2", delta="↓ 12%")
with col3: metric("Uptime", "99.9%", delta="🟢")
with col4: button("Add Model", use_container_width=True)

# Models Table (fetch real API)
st.subheader("🤖 Active Models")
# TODO: requests.get("http://localhost:4000/health") + st.dataframe
st.success("Vision/Reasoning ready! Test /chat/completions.")

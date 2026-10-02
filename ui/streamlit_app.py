"""BankLens UI. Talks to the FastAPI backend (BANKLENS_API, default http://localhost:8000)."""

from __future__ import annotations

import os
from urllib.parse import quote

import requests
import streamlit as st

API = os.getenv("BANKLENS_API", "http://localhost:8000")
BADGE = {
    "SUPPORTED": ("#1a7f37", "SUPPORTED"),
    "REFUTED": ("#cf222e", "REFUTED"),
    "NOT_ENOUGH_INFO": ("#9a6700", "NOT ENOUGH INFO"),
}

st.set_page_config(page_title="BankLens", page_icon="🏦", layout="wide")
st.session_state.setdefault("cost", 0.0)


def call(path: str, payload: dict | None = None):
    try:
        r = (
            requests.post(f"{API}{path}", json=payload, timeout=120)
            if payload is not None
            else requests.get(f"{API}{path}", timeout=30)
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def render_citations(cits: list[dict], key: str) -> None:
    if not cits:
        return
    st.markdown("**Sources**")
    for c in cits:
        label = (
            f"[{c['n']}] {c['bank']} · FY{c['fiscal_year']} · p. {c['page_label']} · {c['section'] or 'n/a'}"
        )
        with st.expander(label):
            left, right = st.columns([1, 1])
            left.markdown(c["snippet"])
            url = f"{API}/page_image/{c['doc_id']}/{c['page_index']}?highlight={quote(c['snippet'][:200])}"
            try:
                img = requests.get(url, timeout=60)
                img.raise_for_status()
                right.image(img.content, caption=f"PDF page {c['page_index']} (printed p. {c['page_label']})")
            except requests.RequestException:
                right.info("Page image unavailable (PDF not found on the API server).")


with st.sidebar:
    st.header("BankLens")
    health = call("/health")
    if health:
        st.caption(f"Config: **{health['config']}** · {health['n_chunks']:,} chunks")
        if health["status"] != "ok":
            st.warning(health.get("error") or "No index built yet.")
    st.metric("Session cost", f"${st.session_state.cost:.4f}")
    st.subheader("Documents")
    for d in call("/documents") or []:
        st.write(f"• {d['bank']} FY{d['fiscal_year']}")

st.title("🏦 BankLens")
st.caption("Grounded Q&A and claim verification over the annual reports of the four largest US banks.")
ask_tab, verify_tab = st.tabs(["Ask", "Verify"])

with ask_tab:
    q = st.text_input("Question", placeholder="What was JPMorgan's CET1 ratio at year-end 2025?", key="q")
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Searching filings…"):
            st.session_state.ask = call("/ask", {"question": q})
        if st.session_state.ask:
            st.session_state.cost += st.session_state.ask["cost_usd"]
    r = st.session_state.get("ask")
    if r:
        if r["abstain"]:
            st.warning("The filings I searched don't contain enough information to answer this.")
        else:
            st.success(r["answer"])
        render_citations(r["citations"], "ask")
        st.caption(f"{r['latency_ms']:.0f} ms · ${r['cost_usd']:.5f}")

with verify_tab:
    claim = st.text_input("Claim", placeholder="Wells Fargo's net income fell in 2025", key="claim")
    if st.button("Verify", type="primary") and claim.strip():
        with st.spinner("Checking the claim against filings…"):
            st.session_state.verify = call("/verify", {"claim": claim})
        if st.session_state.verify:
            st.session_state.cost += st.session_state.verify["cost_usd"]
    v = st.session_state.get("verify")
    if v:
        color, text = BADGE.get(v["verdict"], ("#57606a", v["verdict"]))
        st.markdown(
            f"<span style='background:{color};color:white;padding:6px 14px;border-radius:14px;font-weight:600'>{text}</span>",
            unsafe_allow_html=True,
        )
        st.write(v["explanation"])
        render_citations(v["evidence"], "verify")
        st.caption(f"{v['latency_ms']:.0f} ms · ${v['cost_usd']:.5f}")

st.divider()
st.caption(
    "Supported by the filing = the bank reported it, not independently audited truth. Not financial advice."
)

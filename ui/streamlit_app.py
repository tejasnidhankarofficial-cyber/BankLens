"""BankLens UI. Talks to the FastAPI backend (BANKLENS_API, default http://localhost:8000)."""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("BANKLENS_API", "http://localhost:8000")
BADGE = {
    "SUPPORTED": ("#1a7f37", "SUPPORTED"),
    "REFUTED": ("#cf222e", "REFUTED"),
    "NOT_ENOUGH_INFO": ("#9a6700", "NOT ENOUGH INFO"),
}
EXAMPLE_QUESTIONS = [
    "What was Citigroup's net income for 2025?",
    "What was JPMorgan Chase's Standardized CET1 capital ratio at December 31, 2025?",
    "Which had higher net income for 2025, JPMorgan Chase or Bank of America?",
    "What will JPMorgan's dividend per share be in 2030?",
]
EXAMPLE_CLAIMS = [
    "Wells Fargo's net income fell in 2025.",
    "JPMorgan's net income was $62.5 billion in 2025.",
    "Citigroup is about to be acquired by a foreign bank.",
]

st.set_page_config(page_title="BankLens", page_icon="🏦", layout="wide")
st.session_state.setdefault("cost", 0.0)


def call(path: str, payload: dict | None = None):
    try:
        if payload is not None:
            r = requests.post(f"{API}{path}", json=payload, timeout=180)
        else:
            r = requests.get(f"{API}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


@st.cache_data(show_spinner=False, max_entries=64)
def page_image(doc_id: str, page_index: int, chunk_id: str) -> bytes | None:
    try:
        r = requests.get(f"{API}/page_image/{doc_id}/{page_index}", params={"chunk_id": chunk_id}, timeout=60)
        r.raise_for_status()
        return r.content
    except requests.RequestException:
        return None


def render_citations(cits: list[dict], retrieved: list[dict], key: str) -> None:
    """Clickable citation chips; the selected one shows its passage and the highlighted PDF page."""
    if cits:
        st.markdown("**Sources**")
        labels = [f"[{c['n']}] {c['bank']} FY{c['fiscal_year']} · p.{c['page_label']}" for c in cits]
        choice = st.pills(
            "Citations", labels, default=labels[0], key=f"pill_{key}", label_visibility="collapsed"
        )
        c = cits[labels.index(choice)] if choice in labels else cits[0]
        left, right = st.columns([1, 1])
        with left:
            st.caption(f"{c['bank']} · FY{c['fiscal_year']} · {c['section'] or 'n/a'}")
            st.markdown(c["snippet"])
        with right:
            img = page_image(c["doc_id"], c["page_index"], c["chunk_id"])
            if img:
                st.image(
                    img,
                    caption=f"PDF page {c['page_index']} (printed page {c['page_label']}), passage highlighted",
                )
            else:
                st.info("Page image unavailable (PDF not found on the API server).")
    if retrieved:
        with st.expander("How this answer was found (retrieved passages)"):
            st.dataframe(
                [
                    {
                        "rank": i,
                        "document": r["doc_id"],
                        "pdf page": r["page_index"],
                        "score": round(r["score"], 3),
                    }
                    for i, r in enumerate(retrieved, 1)
                ],
                hide_index=True,
                width="stretch",
            )


def example_buttons(examples: list[str], state_key: str) -> None:
    st.caption("Try an example:")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, key=f"{state_key}_{ex}", width="stretch"):
            st.session_state[state_key] = ex
            st.session_state[f"run_{state_key}"] = True


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
    example_buttons(EXAMPLE_QUESTIONS, "q")
    q = st.text_input("Question", placeholder="What was JPMorgan's CET1 ratio at year-end 2025?", key="q")
    if (st.button("Ask", type="primary") or st.session_state.pop("run_q", False)) and q.strip():
        with st.spinner("Searching filings…"):
            st.session_state.ask = call("/ask", {"question": q})
        if st.session_state.ask:
            st.session_state.cost += st.session_state.ask["cost_usd"]
    r = st.session_state.get("ask")
    if r:
        if r["abstain"]:
            st.warning("The filings searched don't contain enough information to answer this.")
        else:
            st.success(r["answer"])
        render_citations(r["citations"], r.get("retrieved", []), "ask")
        st.caption(f"{r['latency_ms']:.0f} ms · ${r['cost_usd']:.5f}")

with verify_tab:
    example_buttons(EXAMPLE_CLAIMS, "claim")
    claim = st.text_input("Claim", placeholder="Wells Fargo's net income fell in 2025", key="claim")
    if (st.button("Verify", type="primary") or st.session_state.pop("run_claim", False)) and claim.strip():
        with st.spinner("Checking the claim against filings…"):
            st.session_state.verify = call("/verify", {"claim": claim})
        if st.session_state.verify:
            st.session_state.cost += st.session_state.verify["cost_usd"]
    v = st.session_state.get("verify")
    if v:
        color, text = BADGE.get(v["verdict"], ("#57606a", v["verdict"]))
        st.markdown(
            f"<span style='background:{color};color:white;padding:6px 14px;border-radius:14px;"
            f"font-weight:600'>{text}</span>",
            unsafe_allow_html=True,
        )
        st.write(v["explanation"])
        render_citations(v["evidence"], v.get("retrieved", []), "verify")
        st.caption(f"{v['latency_ms']:.0f} ms · ${v['cost_usd']:.5f}")

st.divider()
st.caption(
    "Supported by the filing = the bank reported it, not independently audited truth. Not financial advice."
)

"""
frontend/app.py
---------------
Streamlit UI for the AI Resume Screening System.

Features:
  • Sidebar: configuration (API URL, top-N)
  • Tab 1: Upload resumes + enter job description → run analysis
  • Tab 2: Browse previous results
  • Tab 3: Raw JSON inspector

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# ─── Path fix so imports from project root work ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import STREAMLIT_PAGE_TITLE, STREAMLIT_LAYOUT

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon="🤖",
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }
    .skill-chip-matched { background:#dcfce7; color:#166534; border-radius:12px;
                          padding:2px 10px; margin:2px; display:inline-block; font-size:0.78rem; }
    .skill-chip-missing { background:#fee2e2; color:#991b1b; border-radius:12px;
                          padding:2px 10px; margin:2px; display:inline-block; font-size:0.78rem; }
    .skill-chip-extra   { background:#dbeafe; color:#1e40af; border-radius:12px;
                          padding:2px 10px; margin:2px; display:inline-block; font-size:0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 AI Resume Screening System</h1>
        <p>Classical NLP-powered candidate ranking — no external APIs required</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Helper: render ranking results ──────────────────────────────────────────
def _render_results(result: dict) -> None:
    """Render ranking cards from API response dict."""
    candidates = result.get("top_candidates", [])
    total = result.get("total_resumes_processed", 0)
    st.caption(f"Processed **{total}** resumes · showing top **{len(candidates)}**")

    if not candidates:
        st.warning("No candidates returned. Check that resumes were uploaded correctly.")
        return

    for c in candidates:
        score = c["hybrid_score"]
        if score >= 0.65:
            medal = "🥇"
        elif score >= 0.40:
            medal = "🥈"
        else:
            medal = "🥉"

        with st.expander(
            f"{medal} #{c['rank']}  {c['candidate_name']}  — Hybrid: {score:.1%}",
            expanded=(c["rank"] <= 3),
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hybrid Score",  f"{c['hybrid_score']:.1%}")
            m2.metric("TF-IDF",        f"{c['tfidf_score']:.1%}")
            m3.metric("Semantic",      f"{c['semantic_score']:.1%}")
            m4.metric("Skill Match",   f"{c['skill_match_ratio']:.1%}")

            st.markdown("**✅ Matched Skills**")
            if c["matched_skills"]:
                chips = " ".join(
                    f'<span class="skill-chip-matched">{s}</span>'
                    for s in c["matched_skills"]
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No matched skills found.")

            st.markdown("**❌ Missing Skills**")
            if c["missing_skills"]:
                chips = " ".join(
                    f'<span class="skill-chip-missing">{s}</span>'
                    for s in c["missing_skills"]
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No missing skills — great match!")

            st.markdown("**➕ Extra Skills (bonus)**")
            if c["extra_skills"]:
                chips = " ".join(
                    f'<span class="skill-chip-extra">{s}</span>'
                    for s in c["extra_skills"][:15]
                )
                st.markdown(chips, unsafe_allow_html=True)

            kw = c.get("keyword_overlap", {})
            st.caption(
                f"🔑 Keyword overlap: **{kw.get('common_keyword_count', 0)}** common words "
                f"(JD: {kw.get('jd_keyword_count', 0)}, "
                f"Resume: {kw.get('resume_keyword_count', 0)})"
            )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    api_base = st.text_input("FastAPI base URL", value="http://127.0.0.1:8000")
    top_n    = st.slider("Top N candidates", 1, 30, 10)
    st.divider()
    st.markdown(
        "**Quick Guide:**\n"
        "1. Upload resumes (PDF/TXT)\n"
        "2. Paste a job description\n"
        "3. Click **Analyze** to rank candidates\n"
        "4. Explore explainable results"
    )
    st.divider()
    if st.button("🗑️ Clear uploaded resumes", use_container_width=True):
        try:
            r = requests.delete(f"{api_base}/clear", timeout=5)
            if r.status_code == 200:
                st.success("Session cleared!")
            else:
                st.error(f"Error: {r.text}")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_analyze, tab_results, tab_debug = st.tabs(
    ["🔍 Analyze Candidates", "📊 Browse Results", "🔬 Debug / Raw JSON"]
)

# ── Tab 1: Analyze ───────────────────────────────────────────────────────────
with tab_analyze:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📄 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Drag & drop PDF or TXT résumés",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("⬆️ Upload to server", use_container_width=True):
            with st.spinner("Uploading …"):
                files_payload = [
                    ("files", (f.name, f.read(), "application/octet-stream"))
                    for f in uploaded_files
                ]
                try:
                    resp = requests.post(
                        f"{api_base}/upload_resume",
                        files=files_payload,
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.success(resp.json()["message"])
                    else:
                        st.error(f"Upload failed ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"Cannot reach API — is it running? ({exc})")

        st.divider()
        st.subheader("📝 Job Description")
        jd_text = st.text_area(
            "Paste the full job description below:",
            height=300,
            placeholder=(
                "We are looking for a Senior Data Scientist with 5+ years of experience "
                "in Python, machine learning, TensorFlow, SQL, and cloud platforms (AWS/GCP).\n\n"
                "Responsibilities: build ML pipelines, design data systems, deploy models…"
            ),
        )

    with col2:
        st.subheader("🏆 Ranking Results")

        if st.button("🚀 Analyze & Rank Candidates", use_container_width=True, type="primary"):
            if not jd_text.strip():
                st.warning("Please enter a job description first.")
            else:
                with st.spinner("Running NLP pipeline … this may take a moment."):
                    try:
                        resp = requests.post(
                            f"{api_base}/analyze",
                            data={"job_description": jd_text, "top_n": top_n},
                            timeout=180,
                        )
                        if resp.status_code == 200:
                            result = resp.json()
                            st.session_state["last_result"] = result
                            _render_results(result)
                        else:
                            st.error(f"Analysis failed ({resp.status_code}): {resp.text}")
                    except Exception as exc:
                        st.error(f"API error: {exc}")

        elif "last_result" in st.session_state:
            st.info("Showing last analysis result. Upload new resumes and click Analyze to refresh.")
            _render_results(st.session_state["last_result"])
        else:
            st.info("Upload resumes and click **Analyze** to see results here.")


# ── Tab 2: Browse saved results ───────────────────────────────────────────────
with tab_results:
    st.subheader("📂 Saved Result Files")
    if st.button("🔄 Refresh list", key="refresh_results"):
        st.rerun()

    try:
        resp = requests.get(f"{api_base}/results", timeout=5)
        files = resp.json().get("result_files", [])
    except Exception:
        files = []
        st.warning("Cannot reach API to list saved results.")

    if not files:
        st.info("No saved results yet. Run an analysis first.")
    else:
        selected = st.selectbox("Select a result file:", files)
        if selected and st.button("Load selected result", key="load_result"):
            try:
                resp = requests.get(f"{api_base}/results/{selected}", timeout=10)
                data = resp.json()
                _render_results(data)
            except Exception as exc:
                st.error(f"Error loading result: {exc}")


# ── Tab 3: Debug ──────────────────────────────────────────────────────────────
with tab_debug:
    st.subheader("🔬 Raw JSON Inspector")
    if "last_result" in st.session_state:
        st.json(st.session_state["last_result"])
    else:
        st.info("Run an analysis to inspect the raw JSON response here.")

    st.divider()
    st.subheader("🩺 API Health Check")
    if st.button("Check API status", key="health_btn"):
        try:
            resp = requests.get(f"{api_base}/", timeout=5)
            if resp.status_code == 200:
                st.success("API is online ✅")
                st.json(resp.json())
            else:
                st.error(f"API returned status {resp.status_code}")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

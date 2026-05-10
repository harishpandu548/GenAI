import warnings, os
warnings.filterwarnings("ignore", message="Accessing `__path__`")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import re, streamlit as st
import agents as ag
from urllib.parse import urlparse

st.set_page_config(
    page_title="Deep Research AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@keyframes g { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }

.brand {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399, #a78bfa);
    background-size: 300% 300%; animation: g 5s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.main .block-container { max-width: 740px; padding-top: 2rem; }

/* source cards */
.src-row { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 18px; }
.src-card {
    flex:1; min-width:130px; max-width:185px; padding:10px 12px; border-radius:9px;
    background:#181828; border:1px solid #2a2a40; text-decoration:none; display:block;
    transition: border-color .15s, transform .15s;
}
.src-card:hover { border-color:#7c3aed; transform:translateY(-2px); }
.src-num   { font-size:.6rem; color:#a78bfa; font-weight:700; letter-spacing:.1em; margin-bottom:4px; }
.src-title { font-size:.78rem; color:#ffffff; line-height:1.35;
             display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.src-host  { font-size:.65rem; color:#64748b; margin-top:4px; }

/* section labels */
.sec {
    font-size:.65rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:#94a3b8; border-bottom:1px solid #1e293b; padding-bottom:6px; margin:20px 0 12px;
}

/* score badge */
.score {
    display:inline-block; font-size:.85rem; font-weight:700;
    color:#fbbf24; background:#f59e0b12; border:1px solid #f59e0b40;
    padding:4px 14px; border-radius:999px; margin-bottom:12px;
}

/* run container */
.run-box {
    border: 1px solid #1e293b; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin: 1.2rem 0;
}
.run-topic {
    font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.8rem;
}

#MainMenu, footer, [data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ── session ───────────────────────────────────────────────────────────────────
if "runs" not in st.session_state:
    st.session_state.runs = []

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_sources(text):
    srcs, seen = [], set()
    for block in re.split(r'\n?----\n?', text):
        t = re.search(r'Title:\s*(.+)', block)
        u = re.search(r'URL:\s*(https?://\S+)', block)
        if t and u:
            url = u.group(1).strip().rstrip('.,)')
            if url not in seen:
                seen.add(url)
                srcs.append({"title": t.group(1).strip()[:70], "url": url,
                             "host": urlparse(url).netloc.replace("www.", "")})
    if not srcs:
        for url in re.findall(r'https?://[^\s\)\]"\'<>,;]+', text):
            url = url.rstrip('.,;)')
            if url not in seen:
                seen.add(url)
                h = urlparse(url).netloc.replace("www.", "")
                srcs.append({"title": h, "url": url, "host": h})
    return srcs[:5]

def src_html(srcs):
    cards = "".join(f"""
    <a class="src-card" href="{s['url']}" target="_blank">
        <div class="src-num">SOURCE {i}</div>
        <div class="src-title">{s['title']}</div>
        <div class="src-host">{s['host']}</div>
    </a>""" for i, s in enumerate(srcs, 1))
    return f'<div class="src-row">{cards}</div>'

def get_score(text):
    m = re.search(r'Score:\s*(\d+(?:\.\d+)?/10)', text)
    return m.group(1) if m else None

def render_trace(messages):
    for msg in messages:
        n = type(msg).__name__
        content = str(msg.content) if msg.content else ""
        if "Human" in n:
            st.markdown("**👤 Human Input**")
            st.info(content[:500])
        elif "AI" in n:
            tcs = getattr(msg, "tool_calls", [])
            if tcs:
                for tc in tcs:
                    st.markdown(f"**🔧 Tool Called → `{tc['name']}`**")
                    st.code(str(tc.get("args", "")), language="json")
            elif content.strip():
                st.markdown("**🤖 AI Response**")
                st.success(content[:500])
        elif "Tool" in n:
            st.markdown(f"**📦 Tool Result → `{getattr(msg, 'name', 'tool')}`**")
            st.code(content[:500])

def show_saved_run(run, idx):
    """Render a completed run from history (no streaming)."""
    data = run["data"]
    srcs = parse_sources(data.get("_raw", ""))

    if srcs:
        st.markdown('<div class="sec">Sources</div>', unsafe_allow_html=True)
        st.markdown(src_html(srcs), unsafe_allow_html=True)

    with st.expander("🔍 Search Agent — messages & tool calls"):
        render_trace(run["msgs"]["search"])

    with st.expander("📖 Reader Agent — messages & tool calls"):
        render_trace(run["msgs"]["reader"])

    st.markdown('<div class="sec">Research Report</div>', unsafe_allow_html=True)
    st.markdown(data["report"])
    st.download_button("↓ Export", data["report"], "report.txt", "text/plain",
                       key=f"dl_saved_{idx}")

    st.markdown('<div class="sec">Critic Review</div>', unsafe_allow_html=True)
    score = get_score(data.get("feedback", ""))
    if score:
        st.markdown(f'<span class="score">⭐ {score}</span>', unsafe_allow_html=True)
    st.markdown(data.get("feedback", ""))

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1rem">
    <span class="brand">Deep Research AI</span>
    <p style="color:#475569; font-size:.85rem; margin-top:6px">
        4 agents · Search · Read · Write · Critique
    </p>
</div>
""", unsafe_allow_html=True)

# ── input (always at top, always visible) ────────────────────────────────────
with st.form("research_form", clear_on_submit=True):
    topic = st.text_input("", placeholder="What do you want to research?",
                          label_visibility="collapsed")
    c1, c2 = st.columns([4, 1])
    with c2:
        submitted = st.form_submit_button("🔍 Research", use_container_width=True)

st.divider()

# ── run pipeline if submitted ─────────────────────────────────────────────────
if submitted and topic.strip():

    st.markdown(f'<div class="run-topic">🔬 {topic}</div>', unsafe_allow_html=True)

    data, msgs = {}, {}

    # Step 1 — Search
    with st.status("🔍  Step 1 · Search Agent working…", expanded=True) as s:
        st.caption("Scanning the web for recent sources…")
        agent = ag.build_search_agent()
        r1 = agent.invoke({"messages": [
            ("user", f"Find recent, reliable and detailed information about: {topic}")
        ]})
        data["search_results"] = r1["messages"][-1].content
        data["_raw"] = "\n".join(str(m.content) for m in r1["messages"] if m.content)
        msgs["search"] = r1["messages"]
        srcs = parse_sources(data["_raw"])
        for src in srcs:
            st.markdown(f"↗ [{src['title']}]({src['url']})")
        s.update(label=f"✅  Step 1 · Search Agent — {len(srcs)} sources found",
                 state="complete", expanded=False)

    with st.expander("🔍 Search Agent — messages & tool calls", expanded=False):
        render_trace(r1["messages"])

    if srcs:
        st.markdown(src_html(srcs), unsafe_allow_html=True)

    # Step 2 — Reader
    with st.status("📖  Step 2 · Reader Agent working…", expanded=True) as s:
        st.caption("Reading and extracting content from top source…")
        agent = ag.build_reader_agent()
        r2 = agent.invoke({"messages": [("user",
            f"Based on these search results about '{topic}', "
            f"pick the most relevant URL and scrape it.\n\n{data['search_results'][:800]}"
        )]})
        data["scraped_content"] = r2["messages"][-1].content
        msgs["reader"] = r2["messages"]
        s.update(label="✅  Step 2 · Reader Agent — content extracted",
                 state="complete", expanded=False)

    with st.expander("📖 Reader Agent — messages & tool calls", expanded=False):
        render_trace(r2["messages"])

    # Step 3 — Write (streamed)
    st.markdown('<div class="sec">Research Report</div>', unsafe_allow_html=True)
    rp = st.empty()
    data["report"] = ""
    for chunk in ag.writer_chain.stream({
        "topic": topic,
        "research": f"SEARCH RESULTS:\n{data['search_results']}\n\nCONTENT:\n{data['scraped_content']}"
    }):
        data["report"] += chunk
        rp.markdown(data["report"] + " ▌")
    rp.markdown(data["report"])

    st.download_button("↓ Export report", data["report"], "report.txt", "text/plain",
                       key=f"dl_current_{len(st.session_state.runs)}")

    # Step 4 — Critique (streamed)
    st.markdown('<div class="sec">Critic Review</div>', unsafe_allow_html=True)
    fc = st.empty()
    data["feedback"] = ""
    for chunk in ag.critic_chain.stream({"report": data["report"]}):
        data["feedback"] += chunk
        fc.markdown(data["feedback"] + " ▌")
    fc.markdown(data["feedback"])

    score = get_score(data["feedback"])
    if score:
        st.markdown(f'<span class="score">⭐ {score}</span>', unsafe_allow_html=True)

    st.session_state.runs.append({"topic": topic, "data": data, "msgs": msgs})

# ── past runs ─────────────────────────────────────────────────────────────────
if st.session_state.runs:
    st.markdown(f"**{len(st.session_state.runs)} past research run(s)** — click to expand")
    for i, run in enumerate(reversed(st.session_state.runs)):
        real_idx = len(st.session_state.runs) - 1 - i
        with st.expander(f"📋  {run['topic']}", expanded=False):
            show_saved_run(run, real_idx)

    if st.button("🗑️ Clear history"):
        st.session_state.runs = []
        st.rerun()

# ── empty state ───────────────────────────────────────────────────────────────
elif not submitted:
    st.markdown("""
    <div style="text-align:center; padding:40px 0">
        <div style="font-size:2.2rem">🔬</div>
        <p style="color:#334155; font-size:.9rem; margin-top:12px; line-height:1.8">
            Type any topic above — 4 agents will search,<br>read sources and write a full report.
        </p>
    </div>
    """, unsafe_allow_html=True)

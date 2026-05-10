import warnings, os, sys
warnings.filterwarnings("ignore", message="Accessing `__path__`")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MultiAIAgentSystemProject"))

import re, tempfile, requests, streamlit as st
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="GenAI Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* ── Layout ───────────────────────────────────────────────────── */
.main .block-container { max-width: 920px; margin: 0 auto; padding: 2.5rem 1.5rem 5rem; }

/* ── Animated brand text ──────────────────────────────────────── */
@keyframes gflow { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.brand {
    font-size: 2.6rem; font-weight: 900; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 35%, #34d399 65%, #a78bfa 100%);
    background-size: 300% 300%; animation: gflow 5s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    display: inline-block; line-height: 1.1;
}
.brand-page {
    font-size: 2rem; font-weight: 900; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399, #a78bfa);
    background-size: 300% 300%; animation: gflow 5s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    display: inline-block;
}

/* ── Home hero ────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
}
.hero-sub {
    color: #64748b; font-size: .95rem; margin-top: 10px; font-weight: 400; letter-spacing: 0;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #6366f115; border: 1px solid #6366f130;
    color: #818cf8; font-size: .72rem; font-weight: 600; letter-spacing: .08em;
    padding: 4px 14px; border-radius: 999px; margin-bottom: 18px; text-transform: uppercase;
}

/* ── Tool cards ───────────────────────────────────────────────── */
.tool-card {
    background: #0d0d1c;
    border: 1px solid #1e1e35;
    border-radius: 18px;
    padding: 2rem 1.4rem 1.5rem;
    text-align: center;
    min-height: 210px;
    position: relative; overflow: hidden;
    transition: border-color .25s, transform .25s, box-shadow .25s;
}
.tool-card::after {
    content: ''; position: absolute; inset: 0; border-radius: 18px;
    background: radial-gradient(ellipse at top, var(--glow) 0%, transparent 65%);
    opacity: 0; transition: opacity .3s;
}
.tool-card:hover { transform: translateY(-5px); box-shadow: 0 16px 48px var(--shadow); }
.tool-card:hover::after { opacity: 1; }

.card-research { --glow: #6366f115; --shadow: #6366f120; border-top: 2px solid transparent;
    background-image: linear-gradient(#0d0d1c,#0d0d1c), linear-gradient(90deg,#818cf8,#60a5fa);
    background-origin: border-box; background-clip: padding-box,border-box; }
.card-rag      { --glow: #10b98115; --shadow: #10b98120; border-top: 2px solid transparent;
    background-image: linear-gradient(#0d0d1c,#0d0d1c), linear-gradient(90deg,#34d399,#10b981);
    background-origin: border-box; background-clip: padding-box,border-box; }
.card-city     { --glow: #f59e0b15; --shadow: #f59e0b20; border-top: 2px solid transparent;
    background-image: linear-gradient(#0d0d1c,#0d0d1c), linear-gradient(90deg,#fbbf24,#f59e0b);
    background-origin: border-box; background-clip: padding-box,border-box; }

.card-icon  { font-size: 2.4rem; margin-bottom: 0.9rem; display: block; }
.card-title { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; }
.card-desc  { font-size: .8rem; color: #64748b; line-height: 1.65; }
.card-chip  {
    display: inline-block; margin-top: 14px;
    padding: 3px 12px; border-radius: 999px;
    font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
}
.chip-r { background: #818cf820; color: #818cf8; }
.chip-g { background: #34d39920; color: #34d399; }
.chip-y { background: #fbbf2420; color: #fbbf24; }

/* ── Research pipeline ───────────────────────────────────────── */
.pipeline {
    display: flex; align-items: center; gap: 0;
    margin: 22px 0 28px; padding: 0;
}
.p-step {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 16px; border-radius: 10px;
    font-size: .8rem; font-weight: 600;
    transition: all .3s;
    flex: 1; justify-content: center;
}
.p-sep { color: #1e293b; font-size: 1.2rem; flex-shrink: 0; margin: 0 2px; }
.p-wait { background: #0d0d1c; color: #334155; border: 1px solid #1e1e30; }
.p-run  {
    background: linear-gradient(135deg, #1a0e40, #0e1840);
    color: #a78bfa; border: 1px solid #6366f1;
    box-shadow: 0 0 16px #6366f130;
    animation: pulse-border 2s infinite;
}
.p-done { background: #04201a; color: #34d399; border: 1px solid #065f46; }
@keyframes pulse-border { 0%,100%{box-shadow:0 0 12px #6366f130} 50%{box-shadow:0 0 24px #6366f160} }

/* ── Source cards ────────────────────────────────────────────── */
.src-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
    gap: 10px; margin: 16px 0 26px;
}
.src-card {
    padding: 13px 14px; border-radius: 12px;
    background: #0a0a18; border: 1px solid #1a1a30;
    text-decoration: none; display: block;
    transition: all .2s;
}
.src-card:hover { border-color: #6366f1; transform: translateY(-3px); box-shadow: 0 8px 28px #6366f118; }
.src-num   { font-size: .58rem; color: #6366f1; font-weight: 700; letter-spacing: .14em; margin-bottom: 5px; text-transform: uppercase; }
.src-title { font-size: .78rem; color: #e2e8f0; line-height: 1.45; font-weight: 500;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.src-host  { font-size: .64rem; color: #475569; margin-top: 6px; }

/* ── Report box ──────────────────────────────────────────────── */
.report-box {
    background: #080814; border: 1px solid #16162a; border-radius: 14px;
    padding: 2rem 2.2rem; margin: 10px 0 6px;
    line-height: 1.85; color: #cbd5e1; font-size: .93rem;
}

/* ── Critique box ────────────────────────────────────────────── */
.critique-box {
    background: #0a0a18; border: 1px solid #1a1a30;
    border-left: 3px solid #a78bfa;
    border-radius: 0 12px 12px 12px;
    padding: 1.5rem 1.8rem; margin: 10px 0;
    line-height: 1.8; color: #c4b5fd; font-size: .9rem;
}

/* ── Score badge ─────────────────────────────────────────────── */
.score-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 1rem; font-weight: 800; color: #fbbf24;
    background: #fbbf2410; border: 1px solid #fbbf2445;
    padding: 7px 22px; border-radius: 999px; margin: 14px 0;
}

/* ── Section label ───────────────────────────────────────────── */
.sec {
    font-size: .6rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase;
    color: #475569; border-bottom: 1px solid #1a1a30;
    padding-bottom: 7px; margin: 26px 0 14px;
    display: flex; align-items: center; gap: 8px;
}
.sec::before { content: ''; display: block; width: 3px; height: 12px;
    background: linear-gradient(#6366f1, #818cf8); border-radius: 2px; flex-shrink: 0; }

/* ── RAG answer bubble ───────────────────────────────────────── */
.answer-bubble {
    background: linear-gradient(135deg, #0d0d22, #10102a);
    border: 1px solid #1e1e3a; border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 12px;
    padding: 1.4rem 1.8rem; margin: 12px 0;
    line-height: 1.85; font-size: .93rem; color: #e2e8f0;
}

/* ── City result ─────────────────────────────────────────────── */
.city-box {
    background: #0a0a1a; border: 1px solid #1e1e35; border-radius: 14px;
    padding: 1.5rem 1.8rem; margin: 12px 0;
    line-height: 1.8; font-size: .93rem; color: #e2e8f0;
}

/* ── Inputs ──────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0a0a18 !important; color: #f1f5f9 !important;
    border: 1px solid #252540 !important; border-radius: 11px !important;
    font-size: .92rem !important; padding: 0.65rem 1rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px #6366f120 !important;
    outline: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: #0d0d1e !important; color: #a0aec0 !important;
    border: 1px solid #252540 !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: .85rem !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    border-color: #6366f1 !important; color: #c4b5fd !important;
    transform: translateY(-1px) !important; box-shadow: 0 4px 14px #6366f125 !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    border: none !important; color: #fff !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: .9rem !important; letter-spacing: .02em !important;
    box-shadow: 0 4px 18px #6366f135 !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 6px 24px #6366f155 !important; transform: translateY(-2px) !important;
}

/* ── File uploader ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0a0a18 !important; border: 2px dashed #252540 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }

/* ── Expander ────────────────────────────────────────────────── */
[data-testid="stExpander"] { border: 1px solid #1a1a30 !important; border-radius: 10px !important; }
[data-testid="stExpander"]:hover { border-color: #252545 !important; }

/* ── Hide chrome ─────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stHeader"] { display: none !important; }

/* ── Divider ─────────────────────────────────────────────────── */
hr { border-color: #1a1a30 !important; }

/* ── Info / success / error ──────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("page","home"), ("research_runs",[]),
             ("rag_ready",False), ("city_history",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

def goto(page):
    st.session_state.page = page
    st.rerun()

def back_button():
    if st.button("← Home", key=f"back_{st.session_state.page}"):
        goto("home")

# ── Helpers ────────────────────────────────────────────────────────────────────
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
                             "host": urlparse(url).netloc.replace("www.","")})
    if not srcs:
        for url in re.findall(r'https?://[^\s\)\]"\'<>,;]+', text):
            url = url.rstrip('.,;)')
            if url not in seen:
                seen.add(url)
                h = urlparse(url).netloc.replace("www.","")
                srcs.append({"title": h, "url": url, "host": h})
    return srcs[:5]

def src_html(srcs):
    cards = "".join(f"""
    <a class="src-card" href="{s['url']}" target="_blank" rel="noopener">
        <div class="src-num">SOURCE {i}</div>
        <div class="src-title">{s['title']}</div>
        <div class="src-host">↗ {s['host']}</div>
    </a>""" for i, s in enumerate(srcs, 1))
    return f'<div class="src-grid">{cards}</div>'

def pipeline_html(active):
    steps = [("🔍","Search"), ("📖","Read"), ("✍️","Write"), ("🧐","Critique")]
    parts = []
    for i, (icon, label) in enumerate(steps):
        if i < active:
            cls = "p-done"; prefix = "✓ "
        elif i == active:
            cls = "p-run"; prefix = "⟳ "
        else:
            cls = "p-wait"; prefix = ""
        parts.append(f'<div class="p-step {cls}">{icon} {prefix}{label}</div>')
        if i < len(steps) - 1:
            parts.append('<div class="p-sep">›</div>')
    return '<div class="pipeline">' + "".join(parts) + '</div>'

def get_score(text):
    m = re.search(r'Score:\s*(\d+(?:\.\d+)?/10)', text)
    return m.group(1) if m else None

def render_trace(messages):
    for msg in messages:
        n = type(msg).__name__
        content = str(msg.content) if msg.content else ""
        if "Human" in n:
            st.markdown("**👤 Input**"); st.info(content[:500])
        elif "AI" in n:
            tcs = getattr(msg, "tool_calls", [])
            if tcs:
                for tc in tcs:
                    st.markdown(f"**🔧 Tool → `{tc['name']}`**")
                    st.code(str(tc.get("args","")), language="json")
            elif content.strip():
                st.markdown("**🤖 Response**"); st.success(content[:500])
        elif "Tool" in n:
            st.markdown(f"**📦 Result → `{getattr(msg,'name','tool')}`**")
            st.code(content[:500])

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain.tools import tool as _tool
from tavily import TavilyClient as _Tavily

@_tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "OPENWEATHER_API_KEY not set in .env"
    try:
        url  = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        data = requests.get(url, timeout=8).json()
        if data.get("cod") != 200:
            return f"City '{city}' not found."
        w = data['weather'][0]['description']
        return f"Weather in {city}: {w}, {data['main']['temp']}°C, humidity {data['main']['humidity']}%, feels like {data['main']['feels_like']}°C"
    except Exception as e:
        return f"Weather fetch failed: {e}"

@_tool
def get_news(city: str) -> str:
    """Get latest news about a city."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY not set in .env"
    try:
        results = _Tavily(api_key=api_key).search(
            query=f"Latest news in {city}", max_results=3
        ).get("results", [])
        if not results:
            return f"No news found for {city}."
        return "\n\n".join(
            f"• {r['title']}\n  {r['url']}\n  {r.get('content','')[:150]}"
            for r in results
        )
    except Exception as e:
        return f"News fetch failed: {e}"

@st.cache_resource
def get_city_agent():
    from langgraph.prebuilt import create_react_agent
    from langchain_mistralai import ChatMistralAI
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    return create_react_agent(llm, tools=[get_weather, get_news])


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
def show_home():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⚡ Powered by Mistral AI + LangChain</div>
        <div class="brand">GenAI Studio</div>
        <p class="hero-sub">Three production-grade AI tools — pick one to get started</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="tool-card card-research">
            <span class="card-icon">🔬</span>
            <div class="card-title">Deep Research AI</div>
            <div class="card-desc">4 autonomous agents search the web, read sources, write a structured report and critique it — like having a research team.</div>
            <span class="card-chip chip-r">4 AI Agents</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Launch Research →", key="go_research", use_container_width=True):
            goto("research")

    with c2:
        st.markdown("""
        <div class="tool-card card-rag">
            <span class="card-icon">📄</span>
            <div class="card-title">PDF RAG Chat</div>
            <div class="card-desc">Upload any PDF document and ask questions. Answers are grounded in your document with source citations.</div>
            <span class="card-chip chip-g">Vector Search</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Launch RAG Chat →", key="go_rag", use_container_width=True):
            goto("rag")

    with c3:
        st.markdown("""
        <div class="tool-card card-city">
            <span class="card-icon">🌆</span>
            <div class="card-title">City Intelligence</div>
            <div class="card-desc">Ask about any city — live weather, breaking news, local insights. AI agents call real-time APIs on your behalf.</div>
            <span class="card-chip chip-y">Live Data</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Launch City AI →", key="go_city", use_container_width=True):
            goto("city")


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
def show_research():
    import agents as ag

    col_back, col_title = st.columns([1, 6])
    with col_back:
        back_button()
    with col_title:
        st.markdown('<span class="brand-page">Deep Research AI</span>', unsafe_allow_html=True)
        st.caption("4 autonomous agents · Search → Read → Write → Critique")

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.form("research_form", clear_on_submit=True):
        topic = st.text_input("",
            placeholder="Enter any topic — e.g. 'Impact of AI on healthcare in 2024'",
            label_visibility="collapsed")
        col_a, col_b = st.columns([5, 1])
        with col_b:
            submitted = st.form_submit_button("🔍 Research", use_container_width=True)

    if submitted and topic.strip():
        st.markdown("<hr>", unsafe_allow_html=True)

        # Topic heading
        st.markdown(f"""
        <div style="margin-bottom:6px">
            <div style="font-size:.7rem;color:#6366f1;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px">RESEARCHING</div>
            <div style="font-size:1.5rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.02em">{topic}</div>
        </div>""", unsafe_allow_html=True)

        pipeline_slot = st.empty()
        data, msgs = {}, {}

        try:
            # ── STEP 1: Search ─────────────────────────────────────────
            pipeline_slot.markdown(pipeline_html(0), unsafe_allow_html=True)
            with st.status("**🔍 Search Agent** — scanning the web for sources…", expanded=True) as s1:
                st.caption(f"Query: *Find recent, reliable information about: {topic}*")
                r1 = ag.build_search_agent().invoke({"messages":[
                    ("user", f"Find recent, reliable and detailed information about: {topic}")
                ]})
                data["search_results"] = r1["messages"][-1].content
                data["_raw"] = "\n".join(str(m.content) for m in r1["messages"] if m.content)
                msgs["search"] = r1["messages"]
                srcs = parse_sources(data["_raw"])
                if srcs:
                    st.markdown(f"**Found {len(srcs)} source(s):**")
                    for src in srcs:
                        st.markdown(f"&nbsp;&nbsp;↗ [{src['title']}]({src['url']})")
                s1.update(label=f"✅ **Search Agent** — {len(srcs)} sources indexed", state="complete", expanded=False)

            if srcs:
                st.markdown(src_html(srcs), unsafe_allow_html=True)

            with st.expander("🔍 Search Agent trace", expanded=False):
                render_trace(r1["messages"])

            # ── STEP 2: Read ───────────────────────────────────────────
            pipeline_slot.markdown(pipeline_html(1), unsafe_allow_html=True)
            with st.status("**📖 Reader Agent** — extracting content from sources…", expanded=True) as s2:
                if srcs:
                    st.caption(f"Deep reading: *{srcs[0]['host']}* and related sources")
                r2 = ag.build_reader_agent().invoke({"messages":[("user",
                    f"Based on these search results about '{topic}', "
                    f"pick the most relevant URL and scrape it.\n\n{data['search_results'][:800]}"
                )]})
                data["scraped_content"] = r2["messages"][-1].content
                msgs["reader"] = r2["messages"]
                content_preview = data["scraped_content"][:200].replace('\n', ' ')
                st.caption(f"Extracted: *{content_preview}…*")
                s2.update(label="✅ **Reader Agent** — content extracted", state="complete", expanded=False)

            with st.expander("📖 Reader Agent trace", expanded=False):
                render_trace(r2["messages"])

            # ── STEP 3: Write ──────────────────────────────────────────
            pipeline_slot.markdown(pipeline_html(2), unsafe_allow_html=True)
            st.markdown('<div class="sec">Research Report</div>', unsafe_allow_html=True)

            report_slot = st.empty()
            data["report"] = ""
            with st.spinner("✍️ Writer Agent composing report…"):
                for chunk in ag.writer_chain.stream({
                    "topic": topic,
                    "research": f"SEARCH RESULTS:\n{data['search_results']}\n\nSCRAPED CONTENT:\n{data['scraped_content']}"
                }):
                    data["report"] += chunk
                    report_slot.markdown(
                        f'<div class="report-box">{data["report"]} ▌</div>',
                        unsafe_allow_html=True
                    )
            report_slot.markdown(
                f'<div class="report-box">{data["report"]}</div>',
                unsafe_allow_html=True
            )

            col_dl, _ = st.columns([1, 3])
            with col_dl:
                st.download_button(
                    "↓ Export report", data["report"],
                    f"research_{topic[:30].replace(' ','_')}.txt", "text/plain",
                    key=f"dl_live_{len(st.session_state.research_runs)}"
                )

            # ── STEP 4: Critique ───────────────────────────────────────
            pipeline_slot.markdown(pipeline_html(3), unsafe_allow_html=True)
            st.markdown('<div class="sec">Expert Critique</div>', unsafe_allow_html=True)

            critique_slot = st.empty()
            data["feedback"] = ""
            with st.spinner("🧐 Critic Agent reviewing report quality…"):
                for chunk in ag.critic_chain.stream({"report": data["report"]}):
                    data["feedback"] += chunk
                    critique_slot.markdown(
                        f'<div class="critique-box">{data["feedback"]} ▌</div>',
                        unsafe_allow_html=True
                    )
            critique_slot.markdown(
                f'<div class="critique-box">{data["feedback"]}</div>',
                unsafe_allow_html=True
            )

            # ── Pipeline complete ──────────────────────────────────────
            pipeline_slot.markdown(pipeline_html(4), unsafe_allow_html=True)

            score = get_score(data["feedback"])
            if score:
                st.markdown(f'<div class="score-badge">⭐ Quality Score: {score}</div>', unsafe_allow_html=True)

            st.session_state.research_runs.append({"topic": topic, "data": data, "msgs": msgs})
            st.success("Research complete!")

        except Exception as e:
            st.error(f"Research failed: {e}")

    # ── Past runs ──────────────────────────────────────────────────────────────
    if st.session_state.research_runs:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"**{len(st.session_state.research_runs)} previous research run(s)**")
        for i, run in enumerate(reversed(st.session_state.research_runs)):
            real = len(st.session_state.research_runs) - 1 - i
            with st.expander(f"📋 {run['topic']}"):
                d = run["data"]
                s2 = parse_sources(d.get("_raw",""))
                if s2:
                    st.markdown(src_html(s2), unsafe_allow_html=True)
                st.markdown(f'<div class="report-box">{d.get("report","")}</div>', unsafe_allow_html=True)
                col_x, _ = st.columns([1, 3])
                with col_x:
                    st.download_button("↓ Export", d["report"], "report.txt", "text/plain", key=f"dl_h_{real}")
                sc = get_score(d.get("feedback",""))
                if sc:
                    st.markdown(f'<div class="score-badge">⭐ {sc}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="critique-box">{d.get("feedback","")}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RAG
# ══════════════════════════════════════════════════════════════════════════════
def show_rag():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import ChatPromptTemplate

    col_back, col_title = st.columns([1, 6])
    with col_back:
        back_button()
    with col_title:
        st.markdown('<span class="brand-page">PDF RAG Chat</span>', unsafe_allow_html=True)
        st.caption("Upload any PDF · Ask questions · Get answers grounded in your document")

    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your PDF here or click to browse",
        type="pdf", key="pdf_uploader",
        help="Supports any PDF document up to 200MB"
    )

    if uploaded:
        col_proc, col_info = st.columns([2, 3])
        with col_proc:
            if st.button("⚙️ Process PDF", use_container_width=True):
                with st.status("**Processing PDF…**", expanded=True) as sp:
                    try:
                        st.caption("Loading pages…")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded.read())
                            path = tmp.name
                        docs = PyPDFLoader(path).load()
                        st.caption(f"Loaded {len(docs)} page(s) — splitting into chunks…")
                        chunks = RecursiveCharacterTextSplitter(
                            chunk_size=500, chunk_overlap=50
                        ).split_documents(docs)
                        st.caption(f"Created {len(chunks)} chunks — building vector index…")
                        Chroma.from_documents(
                            documents=chunks,
                            embedding=get_embeddings(),
                            persist_directory="chroma_db_studio"
                        )
                        os.unlink(path)
                        st.session_state.rag_ready = True
                        st.session_state.rag_filename = uploaded.name
                        st.session_state.rag_pages = len(docs)
                        st.session_state.rag_chunks = len(chunks)
                        sp.update(label="✅ PDF indexed — ready to answer questions", state="complete", expanded=False)
                    except Exception as e:
                        sp.update(label="❌ Processing failed", state="error", expanded=False)
                        st.error(f"Failed to process PDF: {e}")
        with col_info:
            if st.session_state.rag_ready:
                st.markdown(f"""
                <div style="background:#0a1a12;border:1px solid #065f46;border-radius:10px;padding:12px 16px;font-size:.82rem;color:#34d399">
                    ✅ <strong>{st.session_state.get('rag_filename','')}</strong><br>
                    <span style="color:#64748b">{st.session_state.get('rag_pages',0)} pages · {st.session_state.get('rag_chunks',0)} chunks indexed</span>
                </div>""", unsafe_allow_html=True)
    else:
        st.session_state.rag_ready = False

    if st.session_state.rag_ready:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom:16px;font-size:.82rem;color:#64748b">
            📄 Active document: <strong style="color:#a78bfa">{st.session_state.get('rag_filename','')}</strong>
        </div>""", unsafe_allow_html=True)

        with st.form("rag_form", clear_on_submit=True):
            query = st.text_input("", placeholder="Ask anything about your document…",
                                  label_visibility="collapsed")
            ask = st.form_submit_button("Ask →", use_container_width=False)

        if ask and query.strip():
            with st.status("**Searching document…**", expanded=True) as sq:
                try:
                    st.caption(f"Query: *{query}*")
                    vs = Chroma(persist_directory="chroma_db_studio",
                                embedding_function=get_embeddings())
                    docs = vs.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k":4,"fetch_k":10,"lambda_mult":0.5}
                    ).invoke(query)
                    st.caption(f"Found {len(docs)} relevant chunk(s) — generating answer…")
                    context = "\n\n".join(d.page_content for d in docs)
                    prompt  = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are a precise document assistant. Answer using only the context provided.\n"
                         "If the answer is not in the context, say: I could not find that in the document."),
                        ("human", "Context:\n{context}\n\nQuestion: {question}")
                    ])
                    llm      = ChatMistralAI(model="mistral-small-2506")
                    response = llm.invoke(prompt.invoke({"context":context,"question":query}))
                    sq.update(label="✅ Answer ready", state="complete", expanded=False)

                    st.markdown('<div class="sec">Answer</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-bubble">{response.content}</div>', unsafe_allow_html=True)

                    with st.expander(f"📄 {len(docs)} source chunk(s) used"):
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**Chunk {i}** — page {d.metadata.get('page','?')}")
                            st.caption(d.page_content)
                            if i < len(docs):
                                st.markdown("---")
                except Exception as e:
                    sq.update(label="❌ Error", state="error", expanded=False)
                    st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#334155">
            <div style="font-size:3rem;margin-bottom:12px">📄</div>
            <div style="font-size:1rem;font-weight:600;color:#475569;margin-bottom:6px">No document loaded</div>
            <div style="font-size:.85rem;color:#334155">Upload a PDF above and click <strong>Process PDF</strong> to start chatting</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CITY
# ══════════════════════════════════════════════════════════════════════════════
def show_city():
    col_back, col_title = st.columns([1, 6])
    with col_back:
        back_button()
    with col_title:
        st.markdown('<span class="brand-page">City Intelligence</span>', unsafe_allow_html=True)
        st.caption("Live weather · Breaking news · AI-powered insights for any city worldwide")

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.form("city_form", clear_on_submit=True):
        query = st.text_input("",
            placeholder="e.g.  What's the weather and latest news in Tokyo?",
            label_visibility="collapsed")
        col_a, col_b = st.columns([5, 1])
        with col_b:
            asked = st.form_submit_button("Ask →", use_container_width=True)

    if asked and query.strip():
        try:
            with st.status("**🤖 City Agent** — calling live APIs…", expanded=True) as sc:
                st.caption("Routing to weather and news tools…")
                agent  = get_city_agent()
                result = agent.invoke({"messages":[("user", query)]})
                msgs   = result["messages"]
                answer = msgs[-1].content
                sc.update(label="✅ **City Agent** — data retrieved", state="complete", expanded=False)

            st.markdown('<div class="sec">City Report</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="city-box">{answer}</div>', unsafe_allow_html=True)

            with st.expander("🔧 Agent tool calls trace", expanded=False):
                render_trace(msgs)

            st.session_state.city_history.append({"q": query, "a": answer})

        except Exception as e:
            st.error(f"City agent error: {e}")

    if st.session_state.city_history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"**{len(st.session_state.city_history)} previous question(s)**")
        for item in reversed(st.session_state.city_history):
            with st.expander(f"💬 {item['q'][:65]}"):
                st.markdown(f'<div class="city-box">{item["a"]}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page
if   page == "home":     show_home()
elif page == "research": show_research()
elif page == "rag":      show_rag()
elif page == "city":     show_city()

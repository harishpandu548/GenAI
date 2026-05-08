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
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@keyframes g { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.brand {
    font-size:2.2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399,#a78bfa);
    background-size:300% 300%; animation:g 5s ease infinite;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.main .block-container { max-width:760px; padding-top:2rem; padding-bottom:3rem; }
.card {
    background:#13131f; border:1px solid #1e1e30; border-radius:14px;
    padding:1.5rem 1.2rem; text-align:center; min-height:175px;
    margin-bottom:0.8rem; transition:border-color .2s,transform .2s;
}
.card:hover { border-color:#7c3aed88; transform:translateY(-3px); }
.card-icon  { font-size:2rem; margin-bottom:0.6rem; }
.card-title { font-size:1rem; font-weight:700; color:#ffffff; margin-bottom:0.5rem; }
.card-desc  { font-size:0.78rem; color:#64748b; line-height:1.55; }
.sec {
    font-size:.62rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
    color:#94a3b8; border-bottom:1px solid #1e293b; padding-bottom:6px; margin:20px 0 12px;
}
.src-row { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 18px; }
.src-card {
    flex:1; min-width:130px; max-width:185px; padding:10px 12px; border-radius:9px;
    background:#181828; border:1px solid #2a2a40; text-decoration:none; display:block;
    transition:border-color .15s,transform .15s;
}
.src-card:hover { border-color:#7c3aed; transform:translateY(-2px); }
.src-num   { font-size:.6rem; color:#a78bfa; font-weight:700; letter-spacing:.1em; margin-bottom:4px; }
.src-title { font-size:.78rem; color:#ffffff; line-height:1.35;
             display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.src-host  { font-size:.65rem; color:#64748b; margin-top:4px; }
.score {
    display:inline-block; font-size:.85rem; font-weight:700;
    color:#fbbf24; background:#f59e0b12; border:1px solid #f59e0b40;
    padding:4px 14px; border-radius:999px; margin-bottom:12px;
}
#MainMenu, footer, [data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("page","home"), ("research_runs",[]),
             ("rag_ready",False), ("city_history",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

def goto(page):
    st.session_state.page = page
    st.rerun()

def back_button():
    if st.button("← Back to Home"):
        goto("home")
    st.divider()

# ── Helpers ───────────────────────────────────────────────────────────────────
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
                seen.add(url); h = urlparse(url).netloc.replace("www.","")
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
            st.markdown("**👤 Human Input**"); st.info(content[:500])
        elif "AI" in n:
            tcs = getattr(msg, "tool_calls", [])
            if tcs:
                for tc in tcs:
                    st.markdown(f"**🔧 Tool Called → `{tc['name']}`**")
                    st.code(str(tc.get("args","")), language="json")
            elif content.strip():
                st.markdown("**🤖 AI Response**"); st.success(content[:500])
        elif "Tool" in n:
            st.markdown(f"**📦 Tool Result → `{getattr(msg,'name','tool')}`**")
            st.code(content[:500])

# ── Cached heavy resources ────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# City agent tools defined at module level (not inside cache fn)
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
            return f"City '{city}' not found. Try a different spelling."
        return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C, humidity {data['main']['humidity']}%"
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
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_home():
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
        <span class="brand">GenAI Studio</span>
        <p style="color:#475569;font-size:.88rem;margin-top:8px">
            3 AI-powered projects — choose one to get started
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card">
            <div class="card-icon">🔬</div>
            <div class="card-title">Deep Research AI</div>
            <div class="card-desc">4 agents search the web, read sources, write a report and critique it</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open →", key="go_research", use_container_width=True):
            goto("research")

    with c2:
        st.markdown("""<div class="card">
            <div class="card-icon">📄</div>
            <div class="card-title">PDF RAG Chat</div>
            <div class="card-desc">Upload any PDF and ask questions — answers grounded in your document</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open →", key="go_rag", use_container_width=True):
            goto("rag")

    with c3:
        st.markdown("""<div class="card">
            <div class="card-icon">🌆</div>
            <div class="card-title">City Assistant</div>
            <div class="card-desc">Get live weather and latest news for any city using AI agents</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open →", key="go_city", use_container_width=True):
            goto("city")


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_research():
    import agents as ag

    back_button()
    st.markdown('<span class="brand">Deep Research AI</span>', unsafe_allow_html=True)
    st.caption("4 agents · Search · Read · Write · Critique")
    st.divider()

    with st.form("research_form", clear_on_submit=True):
        topic = st.text_input("Research topic", placeholder="What do you want to research?",
                              label_visibility="collapsed")
        _, col = st.columns([4, 1])
        with col:
            submitted = st.form_submit_button("🔍 Research", use_container_width=True)

    st.divider()

    if submitted and topic.strip():
        st.markdown(f"### 🔬 {topic}")
        data, msgs = {}, {}

        try:
            with st.status("🔍  Step 1 · Search Agent working…", expanded=True) as s:
                st.caption("Scanning the web…")
                r1 = ag.build_search_agent().invoke({"messages":[
                    ("user", f"Find recent, reliable and detailed information about: {topic}")
                ]})
                data["search_results"] = r1["messages"][-1].content
                data["_raw"] = "\n".join(str(m.content) for m in r1["messages"] if m.content)
                msgs["search"] = r1["messages"]
                srcs = parse_sources(data["_raw"])
                for src in srcs:
                    st.markdown(f"↗ [{src['title']}]({src['url']})")
                s.update(label=f"✅  Step 1 · Found {len(srcs)} sources",
                         state="complete", expanded=False)

            with st.expander("🔍 Search Agent — messages & tool calls"):
                render_trace(r1["messages"])

            if srcs:
                st.markdown(src_html(srcs), unsafe_allow_html=True)

            with st.status("📖  Step 2 · Reader Agent working…", expanded=True) as s:
                st.caption("Reading top source…")
                r2 = ag.build_reader_agent().invoke({"messages":[("user",
                    f"Based on these search results about '{topic}', "
                    f"pick the most relevant URL and scrape it.\n\n{data['search_results'][:800]}"
                )]})
                data["scraped_content"] = r2["messages"][-1].content
                msgs["reader"] = r2["messages"]
                s.update(label="✅  Step 2 · Content extracted",
                         state="complete", expanded=False)

            with st.expander("📖 Reader Agent — messages & tool calls"):
                render_trace(r2["messages"])

            st.markdown('<div class="sec">Research Report</div>', unsafe_allow_html=True)
            rp = st.empty()
            data["report"] = ""
            for chunk in ag.writer_chain.stream({
                "topic": topic,
                "research": f"SEARCH:\n{data['search_results']}\n\nCONTENT:\n{data['scraped_content']}"
            }):
                data["report"] += chunk
                rp.markdown(data["report"] + " ▌")
            rp.markdown(data["report"])
            st.download_button("↓ Export", data["report"], "report.txt", "text/plain",
                               key=f"dl_live_{len(st.session_state.research_runs)}")

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

            st.session_state.research_runs.append({"topic": topic, "data": data, "msgs": msgs})

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.research_runs:
        st.markdown(f"**{len(st.session_state.research_runs)} past run(s)**")
        for i, run in enumerate(reversed(st.session_state.research_runs)):
            real = len(st.session_state.research_runs) - 1 - i
            with st.expander(f"📋 {run['topic']}"):
                d = run["data"]
                s2 = parse_sources(d.get("_raw",""))
                if s2: st.markdown(src_html(s2), unsafe_allow_html=True)
                st.markdown(d.get("report",""))
                st.download_button("↓ Export", d["report"], "report.txt", "text/plain",
                                   key=f"dl_hist_{real}")
                sc = get_score(d.get("feedback",""))
                if sc: st.markdown(f'<span class="score">⭐ {sc}</span>', unsafe_allow_html=True)
                st.markdown(d.get("feedback",""))


# ══════════════════════════════════════════════════════════════════════════════
# RAG PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_rag():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import ChatPromptTemplate

    back_button()
    st.markdown('<span class="brand">PDF RAG Chat</span>', unsafe_allow_html=True)
    st.caption("Upload a PDF · Ask questions · Get answers from your document")
    st.divider()

    # Upload
    uploaded = st.file_uploader("Upload your PDF", type="pdf", key="pdf_uploader")

    if uploaded:
        if st.button("⚙️ Process PDF", use_container_width=True):
            with st.spinner("Processing PDF and building vector store…"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        path = tmp.name
                    docs   = PyPDFLoader(path).load()
                    chunks = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=50
                    ).split_documents(docs)
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=get_embeddings(),
                        persist_directory="chroma_db_studio"
                    )
                    os.unlink(path)
                    st.session_state.rag_ready = True
                    st.session_state.rag_filename = uploaded.name
                    st.success(f"✅ '{uploaded.name}' processed — ask questions below")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
    else:
        st.session_state.rag_ready = False

    # Chat
    if st.session_state.rag_ready:
        st.info(f"📄 Active document: **{st.session_state.get('rag_filename','')}**")
        st.divider()

        with st.form("rag_form", clear_on_submit=True):
            query = st.text_input("Ask a question", placeholder="Ask anything about your PDF…",
                                  label_visibility="collapsed")
            ask = st.form_submit_button("Ask", use_container_width=True)

        if ask and query.strip():
            with st.spinner("Searching document…"):
                try:
                    vs = Chroma(persist_directory="chroma_db_studio",
                                embedding_function=get_embeddings())
                    docs = vs.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k":4,"fetch_k":10,"lambda_mult":0.5}
                    ).invoke(query)
                    context = "\n\n".join(d.page_content for d in docs)
                    prompt  = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are a helpful assistant. Answer using only the context below.\n"
                         "If the answer is not in the context, say: I could not find that in the document."),
                        ("human", "Context:\n{context}\n\nQuestion: {question}")
                    ])
                    llm      = ChatMistralAI(model="mistral-small-2506")
                    response = llm.invoke(prompt.invoke({"context":context,"question":query}))

                    st.markdown('<div class="sec">Answer</div>', unsafe_allow_html=True)
                    st.markdown(response.content)

                    with st.expander("📄 Source chunks used"):
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**Chunk {i}** — page {d.metadata.get('page','?')}")
                            st.caption(d.page_content)
                except Exception as e:
                    st.error(f"Error answering question: {e}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0;color:#334155">
            <div style="font-size:2rem">📄</div>
            <p style="margin-top:10px;font-size:.9rem">Upload a PDF above and click <b>Process PDF</b> to get started</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CITY PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_city():
    back_button()
    st.markdown('<span class="brand">City Assistant</span>', unsafe_allow_html=True)
    st.caption("Ask about weather, news or anything about any city")
    st.divider()

    with st.form("city_form", clear_on_submit=True):
        query = st.text_input("City query", placeholder="e.g. What's the weather and latest news in Mumbai?",
                              label_visibility="collapsed")
        _, col = st.columns([4, 1])
        with col:
            asked = st.form_submit_button("Ask", use_container_width=True)

    if asked and query.strip():
        try:
            with st.status("🤖  City Agent working…", expanded=True) as s:
                st.caption("Calling weather and news tools…")
                agent  = get_city_agent()
                result = agent.invoke({"messages":[("user", query)]})
                msgs   = result["messages"]
                answer = msgs[-1].content
                s.update(label="✅  Done", state="complete", expanded=False)

            with st.expander("🔧 Agent — messages & tool calls"):
                render_trace(msgs)

            st.markdown('<div class="sec">Answer</div>', unsafe_allow_html=True)
            st.markdown(answer)
            st.session_state.city_history.append({"q": query, "a": answer})

        except Exception as e:
            st.error(f"City agent error: {e}")

    if st.session_state.city_history:
        st.divider()
        st.markdown(f"**{len(st.session_state.city_history)} previous question(s)**")
        for item in reversed(st.session_state.city_history):
            with st.expander(f"💬 {item['q'][:60]}"):
                st.markdown(item["a"])


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page
if   page == "home":     show_home()
elif page == "research": show_research()
elif page == "rag":      show_rag()
elif page == "city":     show_city()

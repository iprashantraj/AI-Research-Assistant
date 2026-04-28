"""
app.py

AI Research Assistant — Streamlit UI
"""
import os
import time
import streamlit as st
from core.graph import research_graph

# Ensure safe UTF-8 logging in Windows console
os.environ.setdefault("PYTHONUTF8", "1")

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")

# CSS Styling
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: #F5F5F0 !important;
}

.block-container {
    max-width: 700px;
    margin: auto;
    padding-top: 40px;
}

/* Typography & Contrast Fixes */
h1, h2, h3 {
    color: #1F2421 !important;
    letter-spacing: -0.5px !important;
}

p, label, span {
    color: #2B2B2B !important;
}

[data-testid="stMarkdownContainer"] {
    color: #2B2B2B !important;
    opacity: 1 !important;
}

a {
    color: #6B705C !important;
    font-weight: 500;
}

[data-testid="stAlert"] {
    color: #2B2B2B !important;
}

/* clean input */
textarea {
    border-radius: 10px !important;
    border: 1.5px solid #D6D6C2 !important;
    padding: 12px !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
textarea:focus {
    border: 1.5px solid #6B705C !important;
    outline: none !important;
}

/* clean button */
[data-testid="stButton"] button {
    border-radius: 8px !important;
    background: #6B705C !important;
    color: white !important;
    border: none !important;
    padding: 10px 18px !important;
    margin-top: 16px !important;
    transition: background 0.2s ease !important;
}
[data-testid="stButton"] button p {
    color: white !important;
}
[data-testid="stButton"] button:hover {
    background: #1F2421 !important;
}

/* checkbox visible */
input[type="checkbox"] {
    accent-color: #6B705C !important;
}

</style>
""", unsafe_allow_html=True)

# Initialize state
if "running" not in st.session_state:
    st.session_state.running = False
if "report" not in st.session_state:
    st.session_state.report = ""
if "trials_left" not in st.session_state:
    st.session_state.trials_left = 5
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = 0

st.title("AI Research Assistant")
st.markdown("<p style='color: #6B705C; font-size: 16px; margin-bottom: 4px;'>Calm, structured insights with real sources</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 13px; color: #A5A58D; margin-bottom: 24px;'>Made using LangGraph, Tavily, and Groq</p>", unsafe_allow_html=True)

# UI Inputs
badge_color = "#8da77a"
if st.session_state.trials_left <= 2:
    badge_color = "#c9a86a"
if st.session_state.trials_left == 0:
    badge_color = "#d46a6a"

st.markdown(f'''
<div style="
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    background-color: {badge_color}20;
    color: {badge_color};
    font-size: 13px;
    font-weight: 500;
    border: 1px solid {badge_color}40;
    margin-bottom: 10px;
">
    🧪 {st.session_state.trials_left} trials left
</div>
''', unsafe_allow_html=True)

query = st.text_area("Research Topic", placeholder="e.g. The impact of AI on healthcare...", disabled=st.session_state.running)

run_btn = st.button("Generate Report", disabled=st.session_state.running or st.session_state.trials_left <= 0)
debug_mode = st.checkbox("Show Debug Output", value=False)

# Execution
if run_btn and query.strip():
    current_time = time.time()
    
    if st.session_state.trials_left <= 0:
        st.error("Trial limit reached")
        st.stop()
        
    if current_time - st.session_state.last_run_time < 12:
        remaining = int(12 - (current_time - st.session_state.last_run_time))
        st.warning(f"Wait {remaining}s before next run")
        st.stop()

    st.session_state.running = True
    st.session_state.report = ""
    
    status_placeholder = st.empty()
    debug_expander = st.expander("Pipeline Debug Logs", expanded=True) if debug_mode else st.empty()
    
    initial_state = {
        "query": query.strip(),
        "sub_questions": [],
        "search_results": [],
        "summaries": [],
        "consistency_report": "",
        "final_report": "",
        "use_consistency": False
    }
    
    try:
        # Stream the LangGraph pipeline
        for event in research_graph.stream(initial_state):
            node_name = list(event.keys())[0]
            status_placeholder.info(f"⚙️ Running step: **{node_name}**...")
            
            if debug_mode:
                debug_expander.write(f"--- Completed: {node_name} ---")
                debug_expander.json(event[node_name])
                
            if "final_report" in event[node_name]:
                st.session_state.report = event[node_name]["final_report"]
                
        status_placeholder.success("✅ Research complete!")
        st.session_state.trials_left -= 1
        st.session_state.last_run_time = time.time()
        
    except Exception as e:
        status_placeholder.error(f"❌ Pipeline failed: {str(e)}")
        
    finally:
        st.session_state.running = False

def generate_pdf(text: str) -> bytes:
    import io
    import re
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []
    
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Strip markdown syntax
        clean_line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
        clean_line = clean_line.replace("**", "").replace("*", "")
        
        if line.startswith("# "):
            content.append(Paragraph(f"<b>{clean_line.replace('# ', '')}</b>", styles["Title"]))
        elif line.startswith("## "):
            content.append(Paragraph(f"<b>{clean_line.replace('## ', '')}</b>", styles["Heading1"]))
        elif line.startswith("### "):
            content.append(Paragraph(f"<b>{clean_line.replace('### ', '')}</b>", styles["Heading2"]))
        elif line.startswith("- "):
            clean_line = clean_line.replace("- ", "• ")
            content.append(Paragraph(clean_line, styles["Normal"]))
        else:
            content.append(Paragraph(clean_line, styles["Normal"]))
            
        content.append(Spacer(1, 8))
        
    doc.build(content)
    return buffer.getvalue()


# Results
if st.session_state.report:
    st.divider()
    
    # ── Download Buttons ──
    dl_col1, dl_col2, _ = st.columns([1, 1, 3])
    with dl_col1:
        st.download_button(
            label="📄 Download (Markdown)",
            data=st.session_state.report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with dl_col2:
        pdf_bytes = generate_pdf(st.session_state.report)
        st.download_button(
            label="📑 Download (PDF)",
            data=pdf_bytes,
            file_name="research_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Plain UI Rendering ──
    st.markdown(st.session_state.report)

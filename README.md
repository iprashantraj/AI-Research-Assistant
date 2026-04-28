🧠 AI Research Assistant (LangGraph)

An intelligent multi-step research agent that transforms vague queries into structured, source-backed insights.

Built using LangGraph, Groq, and Tavily, this system plans, searches, filters, and synthesizes research into a clean, professional report with clickable citations.

🚀 Live Demo

 https://ai-research-assistant-prashant.streamlit.app/

![Demo](assets/demo.webp)

✨ What Makes It Different

Unlike basic LLM wrappers, this project uses a structured reasoning pipeline:

🧠 Breaks a vague query into focused sub-questions
🌐 Searches real-time web data
🧹 Filters low-quality sources (no LLM cost)
📊 Synthesizes insights across multiple sources
🔗 Produces a clean report with clickable citations
⚙️ How It Works
User Query
   │
   ▼
 Planner (Groq)
 → Generates 3–5 focused sub-questions

   ▼
 Searcher (Tavily)
 → Fetches real web results

   ▼
 Filter (Python)
 → Removes duplicates & noise

   ▼
 Summarizer (Groq - Batched)
 → Extracts key insights across all sources

   ▼
 Consistency Checker (Optional)
 → Flags contradictions

   ▼
 Report Generator (Python)
 → Builds final markdown report
🛠️ Tech Stack
LangGraph → Agent orchestration
Groq API → Ultra-fast LLM inference (Llama 3)
Tavily API → Real-time search
Streamlit → UI + state management
Python 3.10+
🧠 Key Engineering Highlights
⚡ 1. Batched LLM Calls

Instead of multiple API calls per question, all sources are processed in one single call, reducing:

latency
rate limits
API cost
🧩 2. Hybrid Architecture (LLM + Deterministic)
Filtering → Pure Python
Report generation → Pure Python

➡️ Only critical reasoning uses LLMs

🔁 3. Smart Caching + Cooldown
@st.cache_data prevents duplicate runs
UI cooldown avoids accidental spam
Repeat queries → 0 API calls
🛡️ 4. Fault-Tolerant LLM Layer
Primary model: llama-3.3-70b-versatile
Fallback: llama-3.1-8b-instant
Never crashes due to model failure
🎯 Example Use Cases
Market research
Tech trend analysis
Academic topic breakdown
Competitive analysis
Quick literature summaries
⚙️ Setup
1. Clone repo
git clone https://github.com/yourusername/ai-research-assistant-langgraph.git
cd ai-research-assistant-langgraph
2. Install dependencies
pip install -r requirements.txt
3. Add API keys
cp .env.example .env

Fill in:

GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
4. Run app
streamlit run app.py
⚠️ Free Tier Optimization

Each run uses only:

1 call → Planner
1 call → Summarizer

➡️ Total: 2 API calls per research

Caching ensures repeated queries cost zero.

📌 Future Improvements
Better report visualization (cards, sections)
Export to PDF with formatting
Multi-user session handling
Streaming partial outputs
🧑‍💻 Author

Built by Prashant
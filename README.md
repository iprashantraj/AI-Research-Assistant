# AI Research Assistant (LangGraph)

<!-- GitHub Topics: langgraph, llm, ai-agent, streamlit, groq, tavily -->
<!-- Suggested Repo Name: ai-research-assistant-langgraph -->

An intelligent, multi-step research assistant powered by **LangGraph**, **Groq**, and **Tavily**. This system takes a vague user query, breaks it down into focused sub-questions, searches the web, filters the best sources, and synthesizes a comprehensive, professionally formatted markdown report with clickable citations.

## 🚀 Demo

![Demo](assets/demo.gif)
*(Add `demo.gif` to the `assets/` folder to display a preview here)*

## ✨ Features

- **Multi-Step Agentic Pipeline:** Orchestrated seamlessly via LangGraph.
- **Ultra-Fast & Free Inference:** Powered by Llama-3 70B on the Groq API.
- **Batched LLM Summarization:** Processes multiple questions in a single API call to drastically reduce rate-limit issues.
- **Synthesis-Style Insights:** Generates analytical trends rather than repetitive descriptive summaries.
- **Clickable Citations:** Fully transparent sourcing with clickable markdown links `(Source: [domain.com](URL))`.
- **Intelligent Caching:** Streamlit caching paired with a 12-second execution cooldown to prevent duplicate runs and API quota exhaustion.
- **Interactive Streamlit UI:** Watch the AI plan, search, and synthesize in real-time.

## 🏗️ Architecture

The backend operates as a directed acyclic graph, ensuring predictable, high-quality outputs:

```text
User Query
   │
   ▼
 Planner (Groq: Generates 3-5 sub-questions)
   │
   ▼
 Searcher (Tavily: Retrieves raw search results)
   │
   ▼
 Filter (Pure Python: Removes duplicates/junk, no LLM cost)
   │
   ▼
 Summarizer (Groq: Batched processing for all sub-questions)
   │
   ▼
 Consistency Checker (Groq: Optional toggle to detect contradictions)
   │
   ▼
 Report Generator (Pure Python: Assembles final markdown)
```

## 🛠️ Tech Stack

- **[LangGraph](https://python.langchain.com/docs/langgraph):** Agent and pipeline orchestration.
- **[Groq API](https://groq.com/):** High-speed LLM inference (`llama-3.3-70b-versatile` & `llama-3.1-8b-instant`).
- **[Tavily API](https://tavily.com/):** AI-optimized web search.
- **[Streamlit](https://streamlit.io/):** Interactive frontend and state management.
- **Python 3.10+**

## 🧠 Key Engineering Decisions

1. **Batched Summarization:** The biggest bottleneck in multi-agent research is API rate limiting. This system combines all source texts into one mega-context and extracts answers in a single batched call.
2. **LLM-Free Filtering & Reporting:** Filtering bad URLs and generating the final markdown report are handled by deterministic Python logic, saving tokens and speeding up execution.
3. **Caching + Cooldown:** The UI employs `@st.cache_data` and execution locks. If a user repeats a query, the graph immediately replays from cache with zero API calls.
4. **Robust Fallback Models:** If the primary 70B model encounters a network or rate-limit error, the `generate_text()` helper automatically falls back to an 8B model to ensure the pipeline never crashes.

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-research-assistant-langgraph.git
   cd ai-research-assistant-langgraph
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Copy the example `.env` file and add your API keys (both Groq and Tavily offer generous free tiers).
   ```bash
   cp .env.example .env
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## ⚠️ Notes on Free-Tier Limits
This project is optimized to run smoothly on free-tier API accounts. A standard research run consumes exactly **2 Groq API calls** (1 for planning, 1 for batched summarization). The UI caching prevents accidental duplicate requests from draining your quota.

---

import logging
from pathlib import Path
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

# def llm():         #OpenAI
#     global _llm
#     if _llm is None:
#         if not settings.openai_api_key:
#             raise RuntimeError("OPENAI_API_KEY is missing")
#         _llm = ChatOpenAI(
#             model=settings.openai_model,
#             temperature=0,
#             api_key=settings.openai_api_key,
#         )
#     return _llm
def llm():
    global _llm
    if _llm is None:
        api_key = getattr(settings, "gemini_api_key", None) or getattr(settings, "google_api_key", None)
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing")
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0,
            google_api_key=api_key,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def extract_text_content(value: Any) -> str:
    """Normalize LangChain/Gemini message payloads into a clean plain-text string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if hasattr(value, "content"):
        return extract_text_content(value.content)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                block_type = str(item.get("type", "")).lower()
                if block_type in {"thinking", "reasoning"}:
                    continue
                if item.get("text") is not None:
                    text = extract_text_content(item["text"])
                elif item.get("content") is not None:
                    text = extract_text_content(item["content"])
                else:
                    text = ""
            else:
                text = extract_text_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        block_type = str(value.get("type", "")).lower()
        if block_type in {"thinking", "reasoning"}:
            return ""
        for key in ("text", "content", "answer", "output", "response"):
            if value.get(key) is not None:
                return extract_text_content(value[key])
        return ""
    return str(value).strip()


def _extract_field(result: Any, field: str, allowed: tuple[str, ...], default: str) -> str:
    """Safely read a structured output field from either a Pydantic model or dict."""
    value = result.get(field) if isinstance(result, dict) else getattr(result, field, None)
    return value if value in allowed else default


def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
Return valid JSON like {{"route":"kb"}}.
""")
    route = _extract_field(decision, "route", ("kb", "direct"), "kb")
    return {"source_used": route, "trace": add_trace(state, f"Router → {route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}

def grade_kb(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.
Question: {state['question']}
Private company KB evidence:\n{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    kb_grade = _extract_field(grade, "grade", ("good", "weak"), "weak")
    return {"kb_grade": kb_grade, "trace": add_trace(state, f"KB evidence grade → {kb_grade.upper()}")}

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    items = []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + extract_text_content(result["answer"]))
        items = result.get("results", [])
    elif isinstance(result, list):
        items = result
    else:
        lines.append(extract_text_content(result))

    for item in items:
        if isinstance(item, dict):
            title = str(item.get("title") or "")
            url = str(item.get("url") or "")
            content = str(item.get("content") or "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url or "Web result", "url": url, "type": "web"})
        else:
            lines.append(extract_text_content(item))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
Question: {state['question']}
Web evidence:\n{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}}.
""")
    web_grade = _extract_field(grade, "grade", ("good", "weak"), "weak")
    return {"web_grade": web_grade, "trace": add_trace(state, f"Web evidence grade → {web_grade.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    response = llm().invoke(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state['question']}
""")
    rewritten = extract_text_content(response) or state["current_query"]
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    response = llm().invoke(f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
Question: {state['question']}\n\nPrivate KB:\n{context}
""")
    answer = extract_text_content(response) or (
        "I found relevant internal documentation, but could not generate a formatted response. Please contact the IT help desk."
    )
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = str(d.metadata.get("source") or "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": Path(src).name or src, "url": "", "type": "private_kb"})
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}

def generate_from_web(state: AgentState):
    response = llm().invoke(f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information and may need IT validation before changing company-managed systems.
Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
""")
    answer = extract_text_content(response) or (
        "I found external web results, but could not generate a formatted response. Please review the sources or contact the IT help desk."
    )
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    response = llm().invoke(f"Respond briefly and naturally to: {state['question']}")
    answer = extract_text_content(response) or "Hello! How can I help you with IT support today?"
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    result = agent_graph.invoke(initial)
    result["answer"] = extract_text_content(result.get("answer"))
    return result

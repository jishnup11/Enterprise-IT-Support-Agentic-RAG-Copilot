import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.main import app
from app.rag.state import EvidenceGrade, RouteDecision
from app.rag.workflow import (
    ask,
    direct_answer,
    extract_text_content,
    generate_from_kb,
    generate_from_web,
    rewrite_query,
)


class TestResponseNormalization(unittest.TestCase):
    def test_extract_text_content_variants(self):
        self.assertEqual(extract_text_content("  hello world  "), "hello world")
        self.assertEqual(
            extract_text_content(AIMessage(content="Plain answer")), "Plain answer"
        )
        self.assertEqual(
            extract_text_content(
                AIMessage(content=[{"type": "text", "text": "Gemini block answer"}])
            ),
            "Gemini block answer",
        )
        self.assertEqual(
            extract_text_content(
                AIMessage(
                    content=[
                        {"type": "thinking", "thinking": "Internal chain of thought"},
                        {"type": "text", "text": "Step 1: Open VPN"},
                        {"type": "text", "text": "Step 2: Sign in"},
                    ]
                )
            ),
            "Step 1: Open VPN\nStep 2: Sign in",
        )
        self.assertEqual(extract_text_content({"text": "Dict text"}), "Dict text")
        self.assertEqual(extract_text_content(None), "")

    def test_workflow_nodes_normalize_gemini_blocks(self):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(
            content=[
                {"type": "thinking", "thinking": "reasoning"},
                {"type": "text", "text": "Normalized answer text"},
            ]
        )

        base_state = {
            "question": "How do I connect to VPN?",
            "current_query": "How do I connect to VPN?",
            "kb_docs": [
                Document(
                    page_content="Use NovaSecure VPN.",
                    metadata={"source": r"data\sample_kb\handbook.md"},
                )
            ],
            "web_results": "Title: Vendor\nURL: https://example.com\nContent: Guide",
            "kb_grade": "good",
            "web_grade": "good",
            "answer": "",
            "source_used": "",
            "retry_count": 0,
            "trace": [],
            "citations": [],
        }

        with patch("app.rag.workflow.llm", return_value=fake_llm):
            kb_out = generate_from_kb(base_state)
            web_out = generate_from_web(base_state)
            direct_out = direct_answer(base_state)
            rewrite_out = rewrite_query(base_state)

        self.assertEqual(kb_out["answer"], "Normalized answer text")
        self.assertIsInstance(kb_out["answer"], str)
        self.assertEqual(
            kb_out["citations"],
            [{"title": "handbook.md", "url": "", "type": "private_kb"}],
        )
        self.assertEqual(web_out["answer"], "Normalized answer text")
        self.assertIsInstance(web_out["answer"], str)
        self.assertEqual(direct_out["answer"], "Normalized answer text")
        self.assertIsInstance(direct_out["answer"], str)
        self.assertEqual(rewrite_out["current_query"], "Normalized answer text")
        self.assertIsInstance(rewrite_out["current_query"], str)

    def test_ask_kb_and_web_fallback_paths(self):
        # 1) KB path with Gemini content-block output
        kb_llm = MagicMock()
        kb_router = MagicMock()
        kb_router.invoke.return_value = RouteDecision(route="kb")
        kb_grader = MagicMock()
        kb_grader.invoke.return_value = EvidenceGrade(grade="good")
        kb_llm.with_structured_output.side_effect = [kb_router, kb_grader]
        kb_llm.invoke.return_value = AIMessage(
            content=[{"type": "text", "text": "Use NovaSecure VPN from Software Portal."}]
        )

        fake_retriever = MagicMock()
        fake_retriever.invoke.return_value = [
            Document(
                page_content="Install NovaSecure VPN.",
                metadata={"source": "data/sample_kb/company_it_handbook.md"},
            )
        ]

        with patch("app.rag.workflow.llm", return_value=kb_llm), patch(
            "app.rag.workflow.get_retriever", return_value=fake_retriever
        ):
            kb_result = ask("How do I connect to the company VPN?")

        self.assertEqual(kb_result["source_used"], "private_kb")
        self.assertEqual(
            kb_result["answer"], "Use NovaSecure VPN from Software Portal."
        )
        self.assertIsInstance(kb_result["answer"], str)

        # 2) Web fallback path with Gemini content-block output
        web_llm = MagicMock()
        web_router = MagicMock()
        web_router.invoke.return_value = RouteDecision(route="kb")
        weak_kb_grader = MagicMock()
        weak_kb_grader.invoke.return_value = EvidenceGrade(grade="weak")
        good_web_grader = MagicMock()
        good_web_grader.invoke.return_value = EvidenceGrade(grade="good")
        web_llm.with_structured_output.side_effect = [
            web_router,
            weak_kb_grader,
            good_web_grader,
        ]
        web_llm.invoke.return_value = AIMessage(
            content=[{"type": "text", "text": "External guidance: check Microsoft 365 status."}]
        )

        fake_web = MagicMock()
        fake_web.invoke.return_value = {
            "answer": "Microsoft Teams status is healthy.",
            "results": [
                {
                    "title": "Teams Status",
                    "url": "https://status.cloud.microsoft",
                    "content": "No active outage.",
                }
            ],
        }

        with patch("app.rag.workflow.llm", return_value=web_llm), patch(
            "app.rag.workflow.get_retriever", return_value=fake_retriever
        ), patch("app.rag.workflow.web_search_tool", return_value=fake_web):
            web_result = ask("What is the latest Microsoft Teams outage guidance?")

        self.assertEqual(web_result["source_used"], "web_search")
        self.assertEqual(
            web_result["answer"], "External guidance: check Microsoft 365 status."
        )
        self.assertIsInstance(web_result["answer"], str)
        self.assertEqual(
            web_result["citations"],
            [
                {
                    "title": "Teams Status",
                    "url": "https://status.cloud.microsoft",
                    "type": "web",
                }
            ],
        )

    def test_chat_endpoint_returns_clean_chat_response(self):
        client = TestClient(app)
        fake_result = {
            "answer": [{"type": "text", "text": "Connect via NovaSecure VPN."}],
            "source_used": "private_kb",
            "trace": ["Router → KB", "Answer generation → PRIVATE KB"],
            "citations": [
                {"title": "company_it_handbook.md", "url": "", "type": "private_kb"}
            ],
            "current_query": "How do I connect to the company VPN?",
        }

        with patch("app.api.routes.ask", return_value=fake_result), patch(
            "app.api.routes.write_audit"
        ):
            resp = client.post(
                "/api/chat", json={"question": "How do I connect to the company VPN?"}
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["answer"], "Connect via NovaSecure VPN.")
        self.assertIsInstance(body["answer"], str)
        self.assertEqual(body["source_used"], "private_kb")
        self.assertEqual(
            body["citations"],
            [{"title": "company_it_handbook.md", "url": "", "type": "private_kb"}],
        )

    def test_chat_endpoint_rejects_short_or_blank_question(self):
        client = TestClient(app)
        resp_short = client.post("/api/chat", json={"question": "a"})
        self.assertEqual(resp_short.status_code, 422)

        resp_blank = client.post("/api/chat", json={"question": "   "})
        self.assertEqual(resp_blank.status_code, 422)


if __name__ == "__main__":
    unittest.main()

"""Observability: secret redaction, per-node tracing and the optional LangSmith switch."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from langgraph.errors import GraphInterrupt

import observability
from observability import langsmith_status, make_event, redact, safe_error, traced_node

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# LangSmith switch (environment variables only)
# ---------------------------------------------------------------------------
def test_langsmith_disabled_by_default():
    assert langsmith_status() == "disabled"


@pytest.mark.parametrize("flag", ["LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"])
def test_langsmith_enabled_flags(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert langsmith_status() == "enabled_but_no_api_key"
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_dummy")
    assert langsmith_status() == "enabled"


def test_langsmith_false_means_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_dummy")
    assert langsmith_status() == "disabled"


def test_workflow_runs_with_langsmith_on_and_endpoint_unreachable():
    """Run in a subprocess so LangSmith's global client cannot leak into other tests."""
    script = textwrap.dedent(
        """
        import os
        os.environ.update(GROQ_API_KEY="x", TAVILY_API_KEY="x", OPENWEATHER_API_KEY="x")
        from tests.conftest import FakeLLM, McpStubs
        import backend, mcp_client
        from langgraph.checkpoint.memory import InMemorySaver

        llm = FakeLLM(); backend.llm = llm; mcp_client.llm = llm
        m = McpStubs()
        mcp_client.weather_mcp_search = m.weather
        mcp_client.forecast_mcp_search = m.forecast
        backend.aviation_mcp_call = m.aviation
        backend.tavily_mcp_search = m.tavily
        backend.set_travel_graph(backend.build_travel_graph(InMemorySaver()))
        result = backend.run_travel_agent("Plan a 5-day trip to Goa", "ls-thread")
        assert result["workflow_status"] == "awaiting_approval", result["workflow_status"]
        print("WORKFLOW_OK")
        """
    )
    env = {
        **__import__("os").environ,
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2_not_a_real_key",
        "LANGSMITH_ENDPOINT": "http://127.0.0.1:9",  # nothing listens here
    }
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert "WORKFLOW_OK" in done.stdout, done.stdout + done.stderr


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, secret",
    [
        ("GET https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-abc123XYZ failed", "abc123XYZ"),
        ("url?q=Goa&appid=SECRET123&units=metric", "SECRET123"),
        ("http://api.aviationstack.com/v1/flights?access_key=AVKEY999", "AVKEY999"),
        ("Authorization: Bearer eyJhbGciOi.payload.sig", "eyJhbGciOi"),
        ("postgresql://user:hunter2@db.example.com:5432/app", "hunter2"),
        ("Invalid key gsk_abcdefghijklmnop123", "gsk_abcdefghijklmnop123"),
    ],
)
def test_redact_removes_secrets(text, secret):
    assert secret not in redact(text)
    assert "[REDACTED]" in redact(text)


def test_redact_removes_literal_env_secret(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "some-unusual-key-format-77")
    assert "some-unusual-key-format-77" not in redact("boom: some-unusual-key-format-77")


def test_redact_leaves_normal_text_alone():
    assert redact("Plan a 5-day trip to Goa") == "Plan a 5-day trip to Goa"


def test_safe_error_is_short_and_redacted():
    exc = RuntimeError("failed appid=SECRET123 " + "x" * 1000)
    message = safe_error(exc)
    assert message.startswith("RuntimeError:")
    assert "SECRET123" not in message
    assert len(message) <= 200


# ---------------------------------------------------------------------------
# traced_node
# ---------------------------------------------------------------------------
def test_traced_node_records_a_measured_success_event():
    @traced_node("demo_agent")
    def node(state):
        return {"value": 1}

    update = node({"run_id": "r1"})
    assert update["value"] == 1
    (event,) = update["execution_trace"]
    assert event["node"] == "demo_agent" and event["status"] == "success"
    assert event["run_id"] == "r1" and event["duration_ms"] >= 0


def test_traced_node_uses_trace_metadata_from_the_node():
    @traced_node("demo_agent")
    def node(state):
        return {"_trace": {"status": "degraded", "detail": "used fallback", "cache": "miss"}}

    update = node({"run_id": "r1"})
    (event,) = update["execution_trace"]
    assert (event["status"], event["detail"], event["cache"]) == ("degraded", "used fallback", "miss")
    assert "_trace" not in update  # never leaks into graph state


def test_traced_node_supports_several_events_from_one_node():
    @traced_node("guardrail")
    def node(state):
        return {"_trace": [{"node": "input_validation"}, {"status": "blocked"}]}

    events = node({"run_id": "r1"})["execution_trace"]
    assert [(e["node"], e["status"]) for e in events] == [
        ("input_validation", "success"),
        ("guardrail", "blocked"),
    ]


def test_traced_node_turns_unexpected_exceptions_into_failed_events():
    @traced_node("demo_agent")
    def node(state):
        raise RuntimeError("boom appid=SECRET123")

    (event,) = node({"run_id": "r1"})["execution_trace"]
    assert event["status"] == "failed"
    assert "SECRET123" not in (event["error"] or "")


def test_traced_node_does_not_swallow_langgraph_interrupts():
    """interrupt() works by raising; catching it would break human-in-the-loop."""

    @traced_node("human_approval")
    def node(state):
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        node({"run_id": "r1"})


def test_make_event_rounds_duration():
    assert make_event("r", "n", "success", 12.3456)["duration_ms"] == 12.3

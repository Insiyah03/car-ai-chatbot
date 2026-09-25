"""
Tests for the orchestrator's non-LLM logic: the guardrail keyword
filter, tool dispatch, and the tool-call message builder. These don't
make real API calls - they test what our own code does with inputs
that look like what LiteLLM would hand back.
"""
import json
from types import SimpleNamespace

from app import orchestrator as O
from app.data.loader import load_dataset
from app.persistence.db import init_db
from app.persistence import profiles


def test_guardrail_catches_competitor_mentions():
    reply = O._guardrail_check("I'd also recommend checking Cars24 for more options!")
    assert "cars24" not in reply.lower()
    assert "Cars24" not in reply


def test_guardrail_leaves_clean_replies_untouched():
    clean = "Here are 3 SUVs under AED 60,000."
    assert O._guardrail_check(clean) == clean


def test_tool_call_message_dict_is_minimal_and_json_safe():
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="search_inventory", arguments='{"make": "Toyota"}'),
        )],
    )
    built = O._tool_call_message_dict(message)
    assert built == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_inventory", "arguments": '{"make": "Toyota"}'},
        }],
    }
    json.dumps(built, allow_nan=False)  # must not raise


def test_dispatch_search_inventory_and_records_preference(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    monkeypatch.setattr(O, "settings", SimpleNamespace(db_path=db_path, leads_csv_path=tmp_path / "leads.csv"))

    df = load_dataset()
    call = SimpleNamespace(id="call_1", function=SimpleNamespace(
        name="search_inventory", arguments=json.dumps({"make": "Toyota"}),
    ))
    result = O._dispatch_tool(df, None, "user-x", call)

    assert result["total_matched"] > 0
    saved = profiles.get_profile(db_path, "user-x")
    assert saved["preferences"] == {"make": "Toyota"}


def test_dispatch_unknown_tool_returns_error():
    call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="delete_everything", arguments="{}"))
    result = O._dispatch_tool(load_dataset(), None, "user-x", call)
    assert "error" in result

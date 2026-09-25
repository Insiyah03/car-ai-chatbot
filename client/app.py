"""
Streamlit chat client. Talks to the FastAPI backend over HTTP only -
no LLM calls, no dataset access, no business logic here (that
boundary is deliberate, see Stage 3). Run with:
    uv run streamlit run client/app.py
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

from identity import get_or_create_user_id

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="dubizzle Car Assistant", page_icon="🚗")

# user_id: loaded once from the local persistence file (survives app
# restarts - this is the long-term memory identity).
# session_id: fresh every time Streamlit starts a new session (e.g. a
# hard page refresh), which is exactly the short-term-memory boundary
# we want: same user, new conversation.
if "user_id" not in st.session_state:
    st.session_state.user_id = get_or_create_user_id()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Session info")
    st.caption(f"User ID: `{st.session_state.user_id[:8]}…`")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}…`")
    st.caption(
        "Your User ID persists across app restarts on this machine, so "
        "the assistant can recall your preferences next time. Refreshing "
        "the page starts a new conversation but keeps the same User ID - "
        "useful for demonstrating cross-session memory."
    )
    if st.button("Start a new conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

st.title("🚗 dubizzle Car Assistant")
st.caption("Ask about our inventory, book a test drive, or tell me what you're looking for.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a car, book a test drive…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "user_id": st.session_state.user_id,
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                reply = response.json()["reply"]
            except httpx.ConnectError:
                reply = (
                    "Can't reach the backend - make sure it's running "
                    "(`uv run uvicorn main:app --reload`)."
                )
            except httpx.HTTPStatusError as exc:
                reply = f"The backend returned an error ({exc.response.status_code}). Please try again."
            except httpx.TimeoutException:
                reply = "That took too long to respond - please try again."
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

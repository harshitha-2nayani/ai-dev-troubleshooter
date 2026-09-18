import builtins
import io
from contextlib import redirect_stdout

import streamlit as st


# ============================================================
# LOAD EXISTING AI AGENT
# ============================================================

# Your current app.py starts a terminal chat loop when imported.
# We temporarily make input() return "exit" so the loop stops
# immediately while the RAG agent functions are loaded.

original_input = builtins.input
builtins.input = lambda prompt="": "exit"

try:
    with redirect_stdout(io.StringIO()):
        import app as agent
finally:
    builtins.input = original_input


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Developer Troubleshooter",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Developer Troubleshooting Agent")

st.write(
    "Ask Python and LangChain troubleshooting questions "
    "using a documentation-grounded RAG agent."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🔧 Technologies")

    st.write("• Python")
    st.write("• LangChain")
    st.write("• RAG")
    st.write("• FAISS")
    st.write("• BM25")
    st.write("• Cross-Encoder Reranking")
    st.write("• FLAN-T5")
    st.write("• Query Rewriting")
    st.write("• Conversation Memory")

    st.divider()

    st.write(
        "Answers are generated from the project's "
        "trusted documentation knowledge base."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask a Python or LangChain troubleshooting question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Process using your existing RAG agent
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documentation and generating answer..."
        ):

            answer = agent.process_question(
                question
            )

        st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.messages:

    if st.sidebar.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        agent.conversation_history.clear()

        st.rerun()
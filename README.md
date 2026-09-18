# AI Developer Documentation & Troubleshooting Agent

An AI-powered developer assistant that uses Retrieval-Augmented Generation (RAG) and an agent-style workflow to answer technical troubleshooting questions from a trusted documentation knowledge base.

## 🚀 Project Overview

This project helps developers troubleshoot Python and LangChain-related issues using relevant documentation instead of relying only on the language model's general knowledge.

The system:

- Loads technical documentation
- Splits documents into smaller chunks
- Converts chunks into embeddings
- Stores embeddings in FAISS
- Performs semantic vector search
- Performs BM25 keyword search
- Combines vector and keyword retrieval
- Rewrites user queries
- Reranks retrieved documents
- Filters irrelevant context
- Compresses retrieved context
- Maintains conversation history
- Handles follow-up questions
- Checks retrieval confidence
- Performs recovery searches when retrieval is weak
- Prevents answers for unknown errors
- Generates answers grounded in the documentation
- Displays the sources used

## 🧠 Architecture

```text
User Question
      ↓
Topic Detection
      ↓
Agent Decision
      ↓
┌─────────────────────────┐
│ Conversation Context    │
│          OR             │
│ Documentation Search    │
└─────────────────────────┘
      ↓
Query Rewriting
      ↓
Hybrid Retrieval
 ┌───────────┬───────────┐
 │ FAISS     │ BM25      │
 │ Semantic  │ Keyword   │
 │ Search    │ Search    │
 └───────────┴───────────┘
      ↓
Deduplication
      ↓
Cross-Encoder Reranking
      ↓
Keyword Boosting
      ↓
Topic Filtering
      ↓
Retrieval Confidence Check
      ↓
Recovery Search (if needed)
      ↓
Context Compression
      ↓
Grounded Answer
      ↓
Sources
```

🔍 RAG Pipeline

The project follows a RAG pipeline:

Documents
↓
Document Loading
↓
Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
User Query
↓
Hybrid Retrieval
↓
Reranking
↓
Relevant Context
↓
Grounded Answer

🔎 Hybrid Search

The system combines two retrieval approaches:

FAISS

Used for semantic similarity search.

It helps find documents that have similar meaning to the user's question even when the exact words are different.

BM25

Used for keyword-based retrieval.

It is useful when exact technical terms such as:

ModuleNotFoundError
FAISS
virtual environment
package

appear in the user's question.

Using both approaches improves retrieval coverage.

🔄 Query Rewriting

A local FLAN-T5 model rewrites the user's question into a clearer technical search query.

Example:

User:
What should I check?

↓

Rewritten query:
ModuleNotFoundError troubleshooting what should I check

For follow-up questions, the agent can inherit the previous topic from conversation history.

🎯 Reranking

After the initial retrieval, a Cross-Encoder reranker evaluates the relationship between:

User Question + Retrieved Document

The results are then sorted according to their relevance.

🧹 Context Filtering & Compression

The system removes irrelevant documents and keeps information related to the detected topic.

For example, when troubleshooting:

ModuleNotFoundError

unrelated documentation about:

TypeError
AttributeError
KeyError

is filtered out.

The remaining context is compressed before generating the final answer.

🤖 Agent Workflow

The project uses an agent-style workflow with two tools:

SEARCH_DOCS
CHECK_CONTEXT

The agent decides which action to take based on the user's question.

Example:

User:
What is ModuleNotFoundError?

Agent:
SEARCH_DOCS

Follow-up:

User:
What should I check?

Agent:
CHECK_CONTEXT
↓
Inherit previous topic
↓
SEARCH_DOCS

🛡️ Error Protection

The system avoids using unrelated documentation for unknown errors.

Example:

User:
How do I fix DatabaseConnectionError?

If the documentation does not contain that error, the system responds:

I could not find information about DatabaseConnectionError
in the current documentation.

This reduces the chance of generating unsupported troubleshooting advice.

📊 Retrieval Confidence

The system checks whether retrieved documents contain the expected technical topic.

If retrieval confidence is low:

Initial Search
↓
Confidence Check
↓
FAILED
↓
Recovery Search
↓
Confidence Check
↓
PASSED

This provides a recovery mechanism when the first search does not return strong context.

💬 Conversation Memory

The application stores recent conversations and uses them to understand follow-up questions.

Example:

User:
What is ModuleNotFoundError?

User:
What should I check?

The second question does not explicitly mention the error, so the agent uses conversation history to identify the previous topic.

📚 Knowledge Base

The current knowledge base contains:

data/
├── langchain_docs.txt
└── python_errors.txt

The documentation covers topics such as:

ModuleNotFoundError
ImportError
AttributeError
TypeError
Python packages
Virtual environments
FAISS
RAG
Retrieval
Reranking
Query rewriting
Metadata filtering
Grounding
Troubleshooting practices

🛠️ Technologies Used
Python
LangChain
LangChain Text Splitters
Hugging Face Embeddings
Sentence Transformers
FAISS
BM25
Cross-Encoder
FLAN-T5
RAG
Hybrid Search
Query Rewriting
Reranking
Context Compression
Agent Workflow

📁 Project Structure
ai-dev-troubleshooter/
│
├── data/
│ ├── langchain_docs.txt
│ └── python_errors.txt
│
├── src/
│
├── venv/
│
├── app.py
├── requirements.txt
└── README.md

▶️ How to Run

1. Create the virtual environment
   python -m venv venv
2. Activate the virtual environment
   .\venv\Scripts\Activate.ps1
3. Install dependencies
   pip install -r requirements.txt
4. Run the application
   python app.py
5. Ask a question

Example:

What is ModuleNotFoundError?

You can also ask follow-up questions:

What should I check?

Type:

exit

to stop the application.

🧪 Example
Input
What is FAISS?
Output
FAISS is used for efficient similarity search over vector embeddings.
A RAG application can store document embeddings in FAISS and use
similarity search to retrieve documents that are semantically similar
to a user's question.

🎯 Project Goals

The main goals of this project are:

Build a practical RAG application.
Improve retrieval quality using hybrid search.
Reduce irrelevant context using reranking and filtering.
Handle conversational follow-up questions.
Add agent-style decision making.
Improve reliability using retrieval confidence checks.
Prevent unsupported answers for unknown technical errors.
Generate answers grounded in trusted documentation.

👩‍💻 Author

Built as an AI/RAG learning and portfolio project.

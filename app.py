# ============================================================
# AI Developer Documentation & Troubleshooting Agent
# ============================================================

import re

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. CONVERSATION MEMORY
# ============================================================

conversation_history = []


# ============================================================
# 2. LOAD DOCUMENTS
# ============================================================

documents = []

files = [
    "data/langchain_docs.txt",
    "data/python_errors.txt"
]

for file_path in files:

    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )

    loaded_documents = loader.load()

    documents.extend(
        loaded_documents
    )


print("\n==========================================")
print("DOCUMENT LOADING")
print("==========================================")
print("Documents loaded:", len(documents))


# ============================================================
# 3. ADD METADATA
# ============================================================

for document in documents:

    source = document.metadata.get(
        "source",
        ""
    )

    if "langchain_docs" in source:

        document.metadata["topic"] = "langchain"

    elif "python_errors" in source:

        document.metadata["topic"] = "python"

    else:

        document.metadata["topic"] = "general"


# ============================================================
# 4. SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print("\n==========================================")
print("CHUNKING")
print("==========================================")
print("Total chunks:", len(chunks))


# ============================================================
# 5. CREATE EMBEDDINGS
# ============================================================

print("\n==========================================")
print("LOADING EMBEDDING MODEL")
print("==========================================")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 6. SEPARATE TOPICS
# ============================================================

langchain_chunks = [
    chunk
    for chunk in chunks
    if chunk.metadata.get("topic") == "langchain"
]

python_error_chunks = [
    chunk
    for chunk in chunks
    if chunk.metadata.get("topic") == "python"
]


# ============================================================
# 7. VECTOR STORES
# ============================================================

print("\n==========================================")
print("VECTOR DATABASE")
print("==========================================")

langchain_vector_store = FAISS.from_documents(
    langchain_chunks,
    embeddings
)

python_error_vector_store = FAISS.from_documents(
    python_error_chunks,
    embeddings
)

print(
    "LangChain chunks:",
    len(langchain_chunks)
)

print(
    "Python error chunks:",
    len(python_error_chunks)
)


# ============================================================
# 8. BM25
# ============================================================

def create_bm25(documents_list):

    tokenized_documents = [
        document.page_content.lower().split()
        for document in documents_list
    ]

    return BM25Okapi(
        tokenized_documents
    )


langchain_bm25 = create_bm25(
    langchain_chunks
)

python_error_bm25 = create_bm25(
    python_error_chunks
)


# ============================================================
# 9. LOAD LOCAL LLM
# ============================================================

print("\n==========================================")
print("LOADING LOCAL LLM")
print("==========================================")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)

print(
    "LLM loaded successfully."
)


# ============================================================
# 10. LOAD RERANKER
# ============================================================

print("\n==========================================")
print("LOADING RERANKER")
print("==========================================")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print(
    "Reranker loaded successfully."
)


# ============================================================
# 11. KNOWN ERROR TERMS
# ============================================================

error_terms = [
    "modulenotfounderror",
    "importerror",
    "attributeerror",
    "typeerror",
    "valueerror",
    "keyerror",
    "indexerror",
    "syntaxerror"
]


# ============================================================
# 12. SUPPORTING CONCEPTS
# ============================================================

supporting_concepts = [
    "required package",
    "package is installed",
    "package installed",
    "same virtual environment",
    "virtual environment",
    "installed package",
    "check the import statement",
    "verify that the required component exists",
    "installed package version",
    "package version",
    "current python environment",
    "python environment",
    "troubleshooting process",
    "identify the exact error message",
    "identify the package",
    "relevant documentation"
]


# ============================================================
# 13. CLEAN CONTEXT
# ============================================================

def clean_context(text):

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        lower = stripped.lower()

        if lower in [
            "langchain developer troubleshooting documentation",
            "python error troubleshooting documentation"
        ]:
            continue

        if set(stripped) <= {"=", "-", "_"}:
            continue

        # Remove numbered headings
        if re.match(
            r"^\d+\.\s+.+$",
            stripped
        ):
            continue

        cleaned_lines.append(
            stripped
        )

    return "\n".join(
        cleaned_lines
    )


# ============================================================
# 14. DETECT QUESTION TOPIC
# ============================================================

def detect_question_topic(question):

    question_lower = question.lower()

    # Specific known errors
    for error in error_terms:

        if error in question_lower:
            return error

    # FAISS
    if "faiss" in question_lower:
        return "faiss"

    # LangChain / RAG
    langchain_terms = [
        "langchain",
        "rag",
        "retrieval",
        "retriever",
        "reranking",
        "embedding",
        "vector",
        "vector database"
    ]

    if any(
        term in question_lower
        for term in langchain_terms
    ):
        return "langchain"

    # Python concepts
    python_terms = [
        "python",
        "package",
        "dependency",
        "virtual environment",
        "venv"
    ]

    if any(
        term in question_lower
        for term in python_terms
    ):
        return "python"

    return None


# ============================================================
# 15. EXTRACT UNKNOWN ERROR NAME
# ============================================================

def extract_error_name(question):

    matches = re.findall(
        r"\b[A-Za-z_]+Error\b",
        question
    )

    if matches:
        return matches[0]

    return None


# ============================================================
# 16. CHECK WHETHER ERROR IS KNOWN
# ============================================================

def is_known_error(error_name):

    if not error_name:
        return False

    error_lower = error_name.lower()

    for error in error_terms:

        if error_lower == error:
            return True

    return False


# ============================================================
# 17. QUERY REWRITING
# ============================================================

def rewrite_query(question):

    prompt = f"""
Rewrite the following user question into a clear technical
search query.

Keep the important technical terms.

Do not invent a new error name.

Do not add information that is not present.

User question:

{question}

Search query:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=4,
        early_stopping=True
    )

    rewritten_query = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if not rewritten_query:
        return question

    return rewritten_query


# ============================================================
# 18. PREVIOUS TOPIC
# ============================================================

def get_previous_topic():

    if not conversation_history:
        return None

    for item in reversed(
        conversation_history[-3:]
    ):

        topic = item.get(
            "topic"
        )

        if topic:
            return topic

        previous_question = item.get(
            "question",
            ""
        )

        topic = detect_question_topic(
            previous_question
        )

        if topic:
            return topic

    return None


# ============================================================
# 19. PREVIOUS CONTEXT EVALUATION
# ============================================================

def evaluate_previous_context(question):

    if not conversation_history:
        return False

    question_lower = question.lower()

    previous_item = conversation_history[-1]

    previous_question = previous_item.get(
        "question",
        ""
    )

    previous_answer = previous_item.get(
        "answer",
        ""
    )

    current_topic = detect_question_topic(
        question
    )

    previous_topic = detect_question_topic(
        previous_question
    )

    # Same topic
    if (
        current_topic
        and previous_topic
        and current_topic == previous_topic
    ):
        return True

    # Generic follow-up
    follow_up_phrases = [
        "what should i check",
        "what should i do",
        "what do i check",
        "what about it",
        "what about this",
        "what about that",
        "how do i fix it",
        "how do i fix this",
        "how do i fix that",
        "tell me more",
        "explain more",
        "what next"
    ]

    if any(
        phrase in question_lower
        for phrase in follow_up_phrases
    ):
        return True

    # Reference words
    follow_up_words = [
        "it",
        "this",
        "that",
        "already",
        "then",
        "still",
        "instead",
        "same",
        "another"
    ]

    if any(
        word in question_lower.split()
        for word in follow_up_words
    ):

        if previous_question or previous_answer:
            return True

    return False


# ============================================================
# 20. PREVIOUS ANSWER
# ============================================================

def answer_from_previous_context(question):

    if not conversation_history:
        return ""

    return conversation_history[-1].get(
        "answer",
        ""
    )


# ============================================================
# 21. SEARCH RESOURCES
# ============================================================

def get_search_resources(topic):

    if topic == "langchain":

        return (
            langchain_vector_store,
            langchain_bm25,
            langchain_chunks
        )

    if topic == "python":

        return (
            python_error_vector_store,
            python_error_bm25,
            python_error_chunks
        )

    if topic in error_terms:

        return (
            python_error_vector_store,
            python_error_bm25,
            python_error_chunks
        )

    if topic == "faiss":

        return (
            langchain_vector_store,
            langchain_bm25,
            langchain_chunks
        )

    return (
        None,
        None,
        chunks
    )


# ============================================================
# 22. SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question,
    topic=None
):

    if topic is None:

        topic = detect_question_topic(
            question
        )

    vector_store, bm25, source_chunks = (
        get_search_resources(topic)
    )

    # --------------------------------------------------------
    # Topic-specific search
    # --------------------------------------------------------

    if vector_store is not None:

        vector_results = vector_store.similarity_search(
            question,
            k=5
        )

        tokenized_query = (
            question.lower().split()
        )

        bm25_results = bm25.get_top_n(
            tokenized_query,
            source_chunks,
            n=5
        )

    # --------------------------------------------------------
    # Search both
    # --------------------------------------------------------

    else:

        vector_results = (
            langchain_vector_store.similarity_search(
                question,
                k=5
            )
            +
            python_error_vector_store.similarity_search(
                question,
                k=5
            )
        )

        all_chunks = (
            langchain_chunks
            +
            python_error_chunks
        )

        combined_bm25 = create_bm25(
            all_chunks
        )

        tokenized_query = (
            question.lower().split()
        )

        bm25_results = combined_bm25.get_top_n(
            tokenized_query,
            all_chunks,
            n=5
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined_results = (
        vector_results
        +
        bm25_results
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique_results = []

    seen_content = set()

    for result in combined_results:

        content = result.page_content

        if content not in seen_content:

            unique_results.append(
                result
            )

            seen_content.add(
                content
            )

    return unique_results


# ============================================================
# 23. RERANK
# ============================================================

def rerank_results(
    question,
    results
):

    if not results:
        return []

    rerank_pairs = [
        [
            question,
            result.page_content
        ]
        for result in results
    ]

    scores = reranker.predict(
        rerank_pairs
    )

    ranked_results = sorted(
        zip(
            scores,
            results
        ),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked_results


# ============================================================
# 24. KEYWORD BOOST
# ============================================================

def apply_keyword_boost(
    question,
    ranked_results
):

    question_lower = question.lower()

    boosted_results = []

    for score, result in ranked_results:

        adjusted_score = float(
            score
        )

        document_text = (
            result.page_content.lower()
        )

        for term in error_terms:

            if (
                term in question_lower
                and term in document_text
            ):
                adjusted_score += 5.0

        if (
            "faiss" in question_lower
            and "faiss" in document_text
        ):
            adjusted_score += 5.0

        boosted_results.append(
            (
                adjusted_score,
                result
            )
        )

    boosted_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return boosted_results


# ============================================================
# 25. FILTER CONTEXT BY TOPIC
# ============================================================

def filter_context_by_topic(
    results,
    question,
    topic=None
):

    if not results:
        return []

    question_lower = question.lower()

    # --------------------------------------------------------
    # Determine topic if not supplied
    # --------------------------------------------------------

    if topic is None:
        topic = detect_question_topic(
            question
        )

    # --------------------------------------------------------
    # Specific error from current question
    # --------------------------------------------------------

    requested_error = None

    for error in error_terms:

        if error in question_lower:

            requested_error = error
            break

    # --------------------------------------------------------
    # If current question has a specific error
    # --------------------------------------------------------

    if requested_error:

        filtered = []

        for result in results:

            text_lower = (
                result.page_content.lower()
            )

            if requested_error in text_lower:

                filtered.append(
                    result
                )

        return filtered

    # --------------------------------------------------------
    # IMPORTANT:
    # Inherited specific error from previous conversation
    # --------------------------------------------------------

    if topic in error_terms:

        filtered = []

        for result in results:

            text_lower = (
                result.page_content.lower()
            )

            # Keep the requested error
            if topic in text_lower:

                filtered.append(
                    result
                )

                continue

            # Keep supporting troubleshooting information
            # but do NOT keep unrelated error definitions.
            has_supporting_concept = any(
                concept in text_lower
                for concept in supporting_concepts
            )

            contains_other_error = any(
                error in text_lower
                for error in error_terms
                if error != topic
            )

            if (
                has_supporting_concept
                and not contains_other_error
            ):

                filtered.append(
                    result
                )

        return filtered

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    if (
        topic == "faiss"
        or
        "faiss" in question_lower
    ):

        filtered = []

        for result in results:

            if (
                "faiss"
                in result.page_content.lower()
            ):

                filtered.append(
                    result
                )

        return filtered

    # --------------------------------------------------------
    # General topic
    # --------------------------------------------------------

    return results


# ============================================================
# 26. SELECT RELEVANT CONTEXT
# ============================================================

def select_relevant_context(
    ranked_results,
    question,
    topic=None
):

    if not ranked_results:
        return []

    top_results = [
        result
        for score, result
        in ranked_results[:5]
    ]

    return filter_context_by_topic(
        top_results,
        question,
        topic
    )


# ============================================================
# 27. CONTEXT COMPRESSION
# ============================================================

def compress_context(
    results,
    question,
    topic=None
):

    if not results:
        return []

    question_lower = question.lower()

    if topic is None:
        topic = detect_question_topic(
            question
        )

    compressed = []

    # --------------------------------------------------------
    # Specific error
    # --------------------------------------------------------

    requested_error = None

    for error in error_terms:

        if error in question_lower:

            requested_error = error
            break

    # --------------------------------------------------------
    # Current question contains error
    # --------------------------------------------------------

    if requested_error:

        for result in results:

            text = result.page_content.strip()

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text
            )

            selected = []

            for sentence in sentences:

                if (
                    requested_error
                    in sentence.lower()
                ):

                    selected.append(
                        sentence
                    )

                    continue

                if any(
                    concept in sentence.lower()
                    for concept in supporting_concepts
                ):

                    selected.append(
                        sentence
                    )

            if selected:

                compressed.append(
                    " ".join(
                        selected
                    )
                )

        return compressed

    # --------------------------------------------------------
    # Inherited specific error
    # --------------------------------------------------------

    if topic in error_terms:

        for result in results:

            text = result.page_content.strip()

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text
            )

            selected = []

            for sentence in sentences:

                sentence_lower = sentence.lower()

                if topic in sentence_lower:

                    selected.append(
                        sentence
                    )

                    continue

                if any(
                    concept in sentence_lower
                    for concept in supporting_concepts
                ):

                    selected.append(
                        sentence
                    )

            if selected:

                compressed.append(
                    " ".join(
                        selected
                    )
                )

        return compressed

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    if (
        topic == "faiss"
        or
        "faiss" in question_lower
    ):

        for result in results:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                result.page_content.strip()
            )

            selected = []

            for sentence in sentences:

                if (
                    "faiss"
                    in sentence.lower()
                ):

                    selected.append(
                        sentence
                    )

            if selected:

                compressed.append(
                    " ".join(
                        selected
                    )
                )

        return compressed

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    for result in results:

        cleaned = clean_context(
            result.page_content
        )

        if cleaned:

            compressed.append(
                cleaned
            )

    return compressed


# ============================================================
# 28. RETRIEVAL CONFIDENCE
# ============================================================

def evaluate_retrieval_confidence(
    results,
    question
):

    print(
        "\nEvaluating retrieval quality..."
    )

    if not results:

        print(
            "Confidence check: FAILED"
        )

        return False

    question_lower = question.lower()

    # --------------------------------------------------------
    # UNKNOWN ERROR PROTECTION
    # --------------------------------------------------------

    extracted_error = extract_error_name(
        question
    )

    if extracted_error:

        if not is_known_error(
            extracted_error
        ):

            print(
                "Confidence check: FAILED"
            )

            print(
                "Unknown error:",
                extracted_error
            )

            return False

    # --------------------------------------------------------
    # Specific known error
    # --------------------------------------------------------

    requested_error = None

    for error in error_terms:

        if error in question_lower:

            requested_error = error
            break

    if requested_error:

        matches = 0

        for result in results:

            if (
                requested_error
                in result.page_content.lower()
            ):

                matches += 1

        if matches > 0:

            print(
                "Confidence check: PASSED"
            )

            print(
                "Matching error contexts:",
                matches
            )

            return True

        print(
            "Confidence check: FAILED"
        )

        return False

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    if "faiss" in question_lower:

        matches = 0

        for result in results:

            if (
                "faiss"
                in result.page_content.lower()
            ):

                matches += 1

        if matches > 0:

            print(
                "Confidence check: PASSED"
            )

            print(
                "Matching FAISS contexts:",
                matches
            )

            return True

        print(
            "Confidence check: FAILED"
        )

        return False

    # --------------------------------------------------------
    # General question
    # --------------------------------------------------------

    if len(results) > 0:

        print(
            "Confidence check: PASSED"
        )

        return True

    print(
        "Confidence check: FAILED"
    )

    return False


# ============================================================
# 29. BUILD GROUNDED ANSWER
# ============================================================

def build_grounded_answer(
    context_items,
    question
):

    if not context_items:

        return (
            "I could not find enough information "
            "in the documentation to answer this question."
        )

    question_lower = question.lower()

    # ========================================================
    # UNKNOWN ERROR
    # ========================================================

    extracted_error = extract_error_name(
        question
    )

    if (
        extracted_error
        and
        not is_known_error(
            extracted_error
        )
    ):

        return (
            f"I could not find information about "
            f"{extracted_error} in the current documentation."
        )

    # ========================================================
    # MODULE NOT FOUND ERROR
    # ========================================================

    if "modulenotfounderror" in question_lower:

        cause = None
        checks = []

        for text in context_items:

            lower_text = text.lower()

            if (
                "modulenotfounderror"
                in lower_text
                and
                "means" in lower_text
            ):

                if cause is None:

                    cause = clean_context(
                        text
                    )

            if (
                "required package"
                in lower_text
                or
                "package is installed"
                in lower_text
                or
                "package installed"
                in lower_text
                or
                "same virtual environment"
                in lower_text
                or
                "virtual environment"
                in lower_text
                or
                "check the import statement"
                in lower_text
                or
                "installed package version"
                in lower_text
            ):

                cleaned = clean_context(
                    text
                )

                if cleaned:

                    checks.append(
                        cleaned
                    )

        answer_parts = []

        if cause:

            answer_parts.append(
                f"Cause:\n{cause}"
            )

        if checks:

            answer_parts.append(
                "What to check:\n"
                +
                "\n".join(
                    dict.fromkeys(
                        checks
                    )
                )
            )

        if answer_parts:

            return "\n\n".join(
                answer_parts
            )

    # ========================================================
    # IMPORT ERROR
    # ========================================================

    if "importerror" in question_lower:

        relevant = []

        for text in context_items:

            if "importerror" in text.lower():

                cleaned = clean_context(
                    text
                )

                if cleaned:

                    relevant.append(
                        cleaned
                    )

        if relevant:

            return (
                "ImportError:\n"
                +
                "\n".join(
                    dict.fromkeys(
                        relevant
                    )
                )
            )

    # ========================================================
    # ATTRIBUTE ERROR
    # ========================================================

    if "attributeerror" in question_lower:

        relevant = []

        for text in context_items:

            if (
                "attributeerror"
                in text.lower()
            ):

                cleaned = clean_context(
                    text
                )

                if cleaned:

                    relevant.append(
                        cleaned
                    )

        if relevant:

            return (
                "AttributeError:\n"
                +
                "\n".join(
                    dict.fromkeys(
                        relevant
                    )
                )
            )

    # ========================================================
    # TYPE ERROR
    # ========================================================

    if "typeerror" in question_lower:

        relevant = []

        for text in context_items:

            if (
                "typeerror"
                in text.lower()
            ):

                cleaned = clean_context(
                    text
                )

                if cleaned:

                    relevant.append(
                        cleaned
                    )

        if relevant:

            return (
                "TypeError:\n"
                +
                "\n".join(
                    dict.fromkeys(
                        relevant
                    )
                )
            )

    # ========================================================
    # FAISS
    # ========================================================

    if "faiss" in question_lower:

        relevant = []

        for text in context_items:

            if "faiss" in text.lower():

                cleaned = clean_context(
                    text
                )

                if cleaned:

                    relevant.append(
                        cleaned
                    )

        if relevant:

            return "\n\n".join(
                dict.fromkeys(
                    relevant
                )
            )

    # ========================================================
    # GENERAL TROUBLESHOOTING
    # ========================================================

    troubleshooting_phrases = [
        "what should i check",
        "what should i do",
        "what do i check",
        "how do i fix",
        "how to fix",
        "what next"
    ]

    if any(
        phrase in question_lower
        for phrase in troubleshooting_phrases
    ):

        return (
            "Troubleshooting steps:\n"
            "1. Identify the exact error message.\n"
            "2. Identify the package or component involved.\n"
            "3. Check the relevant documentation.\n"
            "4. Verify the current Python environment.\n"
            "5. Check whether the required dependency is installed.\n"
            "6. Inspect the retrieval results if the application uses RAG.\n"
            "7. Verify that the final answer is supported by the retrieved context."
        )

    # ========================================================
    # GENERAL GROUNDED ANSWER
    # ========================================================

    cleaned = []

    for text in context_items:

        clean_text = clean_context(
            text
        )

        if clean_text:

            cleaned.append(
                clean_text
            )

    if cleaned:

        return "\n\n".join(
            dict.fromkeys(
                cleaned
            )
        )

    return (
        "I could not find enough information "
        "in the documentation to answer this question."
    )


# ============================================================
# 30. SEARCH DOCUMENTATION TOOL
# ============================================================

def search_documentation_tool(
    question,
    topic
):

    print(
        "\nTool called: search_documentation_tool"
    )

    search_question = question

    generic_follow_up_phrases = [
        "what should i check",
        "what should i do",
        "what do i check",
        "what next",
        "how do i fix it",
        "how do i fix this",
        "how do i fix that"
    ]

    question_lower = question.lower()

    is_generic_follow_up = any(
        phrase in question_lower
        for phrase in generic_follow_up_phrases
    )

    if (
        topic
        and
        is_generic_follow_up
    ):

        search_question = (
            f"{topic} troubleshooting "
            f"what should i check"
        )

    tool_query = rewrite_query(
        search_question
    )

    # Preserve explicit topic
    if (
        topic
        and
        topic not in tool_query.lower()
    ):

        tool_query = (
            f"{topic} {tool_query}"
        )

    print(
        "Tool search query:",
        tool_query
    )

    results = search_documents(
        tool_query,
        topic
    )

    print(
        "Tool retrieved documents:",
        len(results)
    )

    return results


# ============================================================
# 31. CONVERSATION CONTEXT TOOL
# ============================================================

def check_conversation_context_tool(
    question
):

    print(
        "\nTool called: check_conversation_context_tool"
    )

    if not conversation_history:

        print(
            "No previous conversation context available."
        )

        return {
            "found": False,
            "answer": ""
        }

    useful = evaluate_previous_context(
        question
    )

    if useful:

        print(
            "Previous conversation contains useful information."
        )

        return {
            "found": True,
            "answer": answer_from_previous_context(
                question
            )
        }

    print(
        "Previous conversation does not contain enough useful information."
    )

    return {
        "found": False,
        "answer": ""
    }


# ============================================================
# 32. TOOL REGISTRY
# ============================================================

agent_tools = {

    "SEARCH_DOCS":
        search_documentation_tool,

    "CHECK_CONTEXT":
        check_conversation_context_tool
}


# ============================================================
# 33. EXECUTE TOOL
# ============================================================

def execute_tool(
    action,
    question,
    topic=None
):

    print(
        "\nAgent selected tool:",
        action
    )

    if action == "SEARCH_DOCS":

        return agent_tools[action](
            question,
            topic
        )

    if action == "CHECK_CONTEXT":

        return agent_tools[action](
            question
        )

    return None


# ============================================================
# 34. VALIDATE TOOL RESULT
# ============================================================

def validate_tool_result(
    action,
    result
):

    print(
        "\nValidating tool result..."
    )

    if action == "SEARCH_DOCS":

        if not result:

            print(
                "Tool result: FAILED"
            )

            return False

        print(
            "Tool result: SUCCESS"
        )

        return True

    if action == "CHECK_CONTEXT":

        if not result:

            print(
                "Tool result: FAILED"
            )

            return False

        if not result.get(
            "found",
            False
        ):

            print(
                "Tool result: FAILED"
            )

            return False

        print(
            "Tool result: SUCCESS"
        )

        return True

    return False


# ============================================================
# 35. AGENT SEARCH WITH RETRY
# ============================================================

def agent_search_with_retry(
    question,
    topic
):

    # --------------------------------------------------------
    # Unknown error guard
    # --------------------------------------------------------

    extracted_error = extract_error_name(
        question
    )

    if (
        extracted_error
        and
        not is_known_error(
            extracted_error
        )
    ):

        print(
            "\nUnknown error detected:",
            extracted_error
        )

        print(
            "Agent will not use unrelated documentation."
        )

        return []

    # --------------------------------------------------------
    # First search
    # --------------------------------------------------------

    results = execute_tool(
        "SEARCH_DOCS",
        question,
        topic
    )

    if not validate_tool_result(
        "SEARCH_DOCS",
        results
    ):

        print(
            "\nSearch tool failed."
        )

        print(
            "Agent recovery: broader search."
        )

        retry_query = (
            f"{question} technical documentation"
        )

        results = search_documents(
            retry_query,
            topic
        )

    if not results:
        return []

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    reranked = rerank_results(
        question,
        results
    )

    # --------------------------------------------------------
    # Keyword boost
    # --------------------------------------------------------

    boosted = apply_keyword_boost(
        question,
        reranked
    )

    # --------------------------------------------------------
    # Topic filter
    # --------------------------------------------------------

    relevant_results = select_relevant_context(
        boosted,
        question,
        topic
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_ok = evaluate_retrieval_confidence(
        relevant_results,
        question
    )

    # --------------------------------------------------------
    # Recovery search
    # --------------------------------------------------------

    if not confidence_ok:

        print(
            "\nRetrieved context was not confident enough."
        )

        print(
            "Agent recovery: SEARCH_DOCS_AGAIN"
        )

        if topic:

            broader_query = (
                f"{topic} troubleshooting "
                f"{question} documentation"
            )

        else:

            broader_query = (
                f"{question} "
                f"technical troubleshooting documentation"
            )

        print(
            "Recovery search query:",
            broader_query
        )

        retry_results = search_documents(
            broader_query,
            topic
        )

        if retry_results:

            retry_reranked = rerank_results(
                broader_query,
                retry_results
            )

            retry_boosted = apply_keyword_boost(
                broader_query,
                retry_reranked
            )

            retry_relevant = select_relevant_context(
                retry_boosted,
                question,
                topic
            )

            retry_confidence = (
                evaluate_retrieval_confidence(
                    retry_relevant,
                    question
                )
            )

            if retry_confidence:

                print(
                    "Recovery successful."
                )

                relevant_results = (
                    retry_relevant
                )

            else:

                print(
                    "Recovery did not find strong context."
                )

                relevant_results = []

    return relevant_results


# ============================================================
# 36. DETERMINE AGENT ACTION
# ============================================================

def decide_action(question):

    question_lower = question.lower()

    # --------------------------------------------------------
    # Known error
    # --------------------------------------------------------

    if any(
        term in question_lower
        for term in error_terms
    ):

        return "SEARCH_DOCS"

    # --------------------------------------------------------
    # Unknown explicit error
    # --------------------------------------------------------

    extracted_error = extract_error_name(
        question
    )

    if extracted_error:

        return "SEARCH_DOCS"

    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    follow_up_phrases = [
        "what should i check",
        "what should i do",
        "what do i check",
        "what about",
        "how about",
        "tell me about",
        "what next",
        "how do i fix it",
        "how do i fix this",
        "how do i fix that"
    ]

    follow_up_words = [
        "it",
        "this",
        "that",
        "already",
        "then",
        "still",
        "instead",
        "same",
        "another"
    ]

    is_follow_up = (
        any(
            phrase in question_lower
            for phrase in follow_up_phrases
        )
        or
        any(
            word in question_lower.split()
            for word in follow_up_words
        )
    )

    if (
        is_follow_up
        and
        conversation_history
    ):

        return "CHECK_CONTEXT"

    # --------------------------------------------------------
    # Troubleshooting
    # --------------------------------------------------------

    troubleshooting_words = [
        "how do i fix",
        "how to fix",
        "why",
        "error",
        "issue",
        "problem",
        "not working",
        "fails",
        "failure"
    ]

    if any(
        word in question_lower
        for word in troubleshooting_words
    ):

        return "SEARCH_DOCS"

    # --------------------------------------------------------
    # Known topics
    # --------------------------------------------------------

    if detect_question_topic(
        question
    ):

        return "SEARCH_DOCS"

    return "ANSWER"


# ============================================================
# 37. COMPLETE AGENT WORKFLOW
# ============================================================

def run_agent_workflow(
    question,
    topic
):

    action = decide_action(
        question
    )

    print(
        "\n=========================================="
    )

    print(
        "Workflow action:",
        action
    )

    # ========================================================
    # CHECK CONTEXT
    # ========================================================

    if action == "CHECK_CONTEXT":

        context_result = execute_tool(
            "CHECK_CONTEXT",
            question
        )

        if validate_tool_result(
            "CHECK_CONTEXT",
            context_result
        ):

            previous_topic = get_previous_topic()

            current_topic = detect_question_topic(
                question
            )

            # Generic follow-up
            if (
                previous_topic
                and
                current_topic is None
            ):

                print(
                    "\nAgent inherited previous topic:",
                    previous_topic
                )

                topic = previous_topic

                action = "SEARCH_DOCS"

            else:

                return context_result[
                    "answer"
                ]

        else:

            print(
                "\nContext tool failed."
            )

            print(
                "Agent recovery: SEARCH_DOCS"
            )

            action = "SEARCH_DOCS"

    # ========================================================
    # ANSWER
    # ========================================================

    if action == "ANSWER":

        if conversation_history:

            return conversation_history[-1].get(
                "answer",
                "I need more information to answer this question."
            )

        return (
            "I need more information from the documentation "
            "to answer this question."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    if action == "SEARCH_DOCS":

        results = agent_search_with_retry(
            question,
            topic
        )

        if not results:

            extracted_error = extract_error_name(
                question
            )

            if (
                extracted_error
                and
                not is_known_error(
                    extracted_error
                )
            ):

                return (
                    f"I could not find information about "
                    f"{extracted_error} in the current documentation."
                )

            return (
                "I could not find enough relevant information "
                "in the documentation to answer this question."
            )

        # ----------------------------------------------------
        # Compress
        # ----------------------------------------------------

        compressed_context = compress_context(
            results,
            question,
            topic
        )

        # ----------------------------------------------------
        # Grounded answer
        # ----------------------------------------------------

        answer = build_grounded_answer(
            compressed_context,
            question
        )

        # ----------------------------------------------------
        # Context display
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "RELEVANT CONTEXTS"
        )

        print(
            "=========================================="
        )

        for i, result in enumerate(
            results
        ):

            print(
                f"\nContext {i + 1}:"
            )

            print(
                clean_context(
                    result.page_content
                )
            )

        # ----------------------------------------------------
        # Final answer
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "FINAL ANSWER:"
        )

        print(
            "=========================================="
        )

        print(
            answer
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "SOURCES USED"
        )

        print(
            "=========================================="
        )

        shown_sources = set()

        for result in results:

            source = result.metadata.get(
                "source",
                "Unknown source"
            )

            if source not in shown_sources:

                print(
                    source
                )

                shown_sources.add(
                    source
                )

        return answer

    return (
        "The agent could not determine what action to take."
    )


# ============================================================
# 38. PROCESS QUESTION
# ============================================================

def process_question(question):

    # --------------------------------------------------------
    # Detect topic
    # --------------------------------------------------------

    topic = detect_question_topic(
        question
    )

    # --------------------------------------------------------
    # Inherit previous topic
    # --------------------------------------------------------

    if (
        topic is None
        and
        conversation_history
        and
        evaluate_previous_context(
            question
        )
    ):

        previous_topic = get_previous_topic()

        if previous_topic:

            topic = previous_topic

    print(
        "\n=========================================="
    )

    print(
        "USER QUESTION:"
    )

    print(
        question
    )

    print(
        "Detected topic:",
        topic
    )

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    answer = run_agent_workflow(
        question,
        topic
    )

    # --------------------------------------------------------
    # Save memory
    # --------------------------------------------------------

    conversation_history.append(
        {
            "question": question,
            "answer": answer,
            "topic": topic,
            "query": rewrite_query(question)
        }
    )

    # --------------------------------------------------------
    # Keep last 10 conversations
    # --------------------------------------------------------

    if len(
        conversation_history
    ) > 10:

        conversation_history.pop(
            0
        )

    return answer


# ============================================================
# 39. CHAT LOOP
# ============================================================

print(
    "\n=========================================="
)

print(
    "AI DEVELOPER TROUBLESHOOTING AGENT"
)

print(
    "=========================================="
)

print(
    "Type your technical question."
)

print(
    "Type 'exit' to stop."
)


while True:

    question = input(
        "\nEnter your question: "
    ).strip()

    if question.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break

    if not question:

        print(
            "Please enter a question."
        )

        continue

    process_question(
        question
    )
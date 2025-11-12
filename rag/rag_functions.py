from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def extend_chathistory(chat_history, user_input, llm_answer):
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=llm_answer)
    ])
    return chat_history


def get_retriver(n_search_kwargs=2):
    PG_DSN = os.getenv("DB_DSN")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Connect to the same vectorstore / collection
    vectorstore = PGVector(
        connection=PG_DSN,
        embeddings=embeddings,
        collection_name="chunks"
    )

    # ✅ Directly test the vectorstore
    docs = vectorstore.similarity_search(
        "When does a player win in Terraforming Mars?", k=3)
    print("Docs retrieved:", len(docs))
    for i, d in enumerate(docs):
        print(f"DOC {i}:", d.metadata.get("source"),
              "|", d.page_content[:150], "\n---")

    # ✅ Or, if you have a retriever:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents("When does a player win in Catan?")
    print("Retriever returned:", len(docs))

    # Build retriever for top-k chunk retrieval
    return vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})

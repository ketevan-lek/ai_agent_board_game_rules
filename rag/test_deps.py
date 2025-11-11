from langchain_postgres import PGVector
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv


if __name__ == "__main__":

    load_dotenv()

    PG_DSN = os.getenv("DB_DSN")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = PGVector(
        connection=PG_DSN,
        embeddings=embeddings,
        collection_name="chunks"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

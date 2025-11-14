from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_history_aware_retriever
from rag_prompts import get_history_aware_message, get_qa_message
import rag_functions as rf

if __name__ == "__main__":
    llm = init_chat_model("gpt-4o-mini")
    retriever = rf.get_retriver()

    # prompts
    context_q_prompt = get_history_aware_message()
    qa_prompt = get_qa_message(add_context=True)

    # chains
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, context_q_prompt)
    document_prompt = PromptTemplate.from_template(
        "From {source}: \n{page_content}"
    )

    # 4) Build the chain that stuffs docs (rendered with doc_name) into {context}
    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        # <-- inject doc_name per document
        document_prompt=document_prompt,
        document_separator="\n\n---\n\n"              # optional, nicer delimiting
    )
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)

    chat_history = []

    print("Type 'exit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        response = rag_chain.invoke(
            {"input": user_input, "chat_history": chat_history})
        answer = response["answer"]

        # update chat history
        chat_history = rf.extend_chathistory(chat_history, user_input, answer)

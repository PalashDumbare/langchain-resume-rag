from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.prompt import get_prompt
from config import LLM_MODEL
from src.history import get_history


def _extract_question(values):
    if isinstance(values, dict):
        question = values.get("question")
    else:
        question = values

    if isinstance(question, list):
        return question[0].content if question else ""
    return question or ""

def _get_context(values, retriever):
    print("Retrieving context for question:", values)
    question = _extract_question(values)
    return retriever.invoke(question)

def get_rag_chain(retriever):
    model = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt = get_prompt()

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda values: _get_context(values, retriever)),
            question=RunnableLambda(_extract_question)
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

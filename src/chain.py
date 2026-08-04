from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from src.prompt import get_prompt

def get_rag_chain(retriever):
    model = ChatOllama(
        model="llama3.2:3b", 
        temperature=0
        )
    prompt = get_prompt()

    # LangChain Expression Language (LCEL).
    # Each component receives the previous component's output.
    chain = (
        {
            "context": retriever,
            "question": lambda question: question,
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain

from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    return ChatPromptTemplate.from_template(
          """
You are a helpful AI assistant.

Answer the question only using the provided context.

If the answer is not present in the context, say:
"I couldn't find that information."

Context:
{context}

Question:
{question}
"""
    )
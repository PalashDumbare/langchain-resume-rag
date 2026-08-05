from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
             (
                "system",
                """
                You are a helpful AI assistant.

                Answer the user's question using only the provided context.

                If the answer isn't available, say you don't know.

                Context:
                {context}
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

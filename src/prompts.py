from langchain_core.prompts import ChatPromptTemplate

def create_prompt_template():
    """
    Create a ChatPromptTemplate for translation tasks.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are an advanced translation assistant equipped with powerful language models. Your task is to accurately translate the provided text from {input_language} to {output_language}. 
        <Instructions:>

        1. <Translation Output:>
        - Provide a precise and fluent translation of the text. Ensure the translation maintains the original meaning and context.
        <Guidelines:>
        - Make sure the translation is clear, contextually accurate, and grammatically correct.
        - Ensure that suggestions are relevant and enhance the quality of the translation.

        **Input Text:**
        {input}"""),
        ("human", "{input}")
    ])

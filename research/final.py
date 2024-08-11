import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

load_dotenv()

class TranslatorApp:
    def __init__(self, groq_api_key, google_api_key, openai_api_key):
        self.groq_chain = None
        self.google_chain = None
        self.openai_chain = None
        
        try:
            self.groq_chatbot = self._initialize_chatbot(groq_api_key, "llama3-70b-8192", ChatGroq)
            self.groq_chain = self._create_translation_chain(self.groq_chatbot)
        except Exception as e:
            print(f"Initialization Error for Groq: {e}")

        try:
            self.google_chatbot = self._initialize_chatbot(google_api_key, "gemini-1.5-pro", ChatGoogleGenerativeAI)
            self.google_chain = self._create_translation_chain(self.google_chatbot)
        except Exception as e:
            print(f"Initialization Error for Google: {e}")

        try:
            self.openai_chatbot = self._initialize_chatbot(openai_api_key, "gpt-4o", ChatOpenAI)
            self.openai_chain = self._create_translation_chain(self.openai_chatbot)
        except Exception as e:
            print(f"Initialization Error for OpenAI: {e}")

    def _initialize_chatbot(self, api_key, model_name, chatbot_class):
        if not api_key:
            raise ValueError(f"API key for {chatbot_class.__name__} is missing.")

        init_params = {
            "api_key": api_key,
            "temperature": 0,
            "max_tokens": None,
            "timeout": None,
            "max_retries": 2,
        }

        # Adjust the parameter name based on the chatbot class
        if chatbot_class == ChatGoogleGenerativeAI:
            init_params["model"] = model_name
        else:
            init_params["model_name"] = model_name

        try:
            return chatbot_class(**init_params)
        except TypeError as e:
            raise TypeError(f"Error initializing {chatbot_class.__name__}: {e}")

    def _create_translation_chain(self, chatbot):
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
                    ("human", "{input}")
                ]
            )
            return prompt | chatbot | StrOutputParser()
        except Exception as e:
            print(f"Error creating translation chain: {e}")
            raise

    def _translate(self, chain, input_language, output_language, input_text):
        try:
            if chain is None:
                return "Translation chain not available."
            return chain.invoke({
                "input_language": input_language,
                "output_language": output_language,
                "input": input_text,
            })
        except Exception as e:
            return f"Error: {e}"

    def perform_translations(self, model_name, input_language, output_language, input_text):
        chain_map = {
            "Groq": self.groq_chain,
            "Google": self.google_chain,
            "OpenAI": self.openai_chain,
        }
        selected_chain = chain_map.get(model_name)
        if selected_chain:
            return self._translate(selected_chain, input_language, output_language, input_text)
        return "Invalid model selected."

class GradioInterface:
    def __init__(self, translator_app):
        self.translator = translator_app
        self._create_interface()

    def _create_interface(self):
        def translate_text(model_name, input_text, input_language, output_language):
            return self.translator.perform_translations(model_name, input_language, output_language, input_text)

        self.interface = gr.Interface(
            fn=translate_text,
            inputs=[
                gr.Dropdown(
                    choices=["Groq", "Google", "OpenAI"],
                    label="Choose Translation Model",
                    value="Groq"
                ),
                gr.Textbox(label="Input Text"),
                gr.Textbox(label="Input Language", value="English"),
                gr.Textbox(label="Output Language", value="German"),
            ],
            outputs=[
                gr.Textbox(label="Translation"),
            ],
            title="Multilingual Translator",
            description="Translate text between different languages using your chosen chatbot model."
        )

    def launch(self):
        self.interface.launch(share=True)  # Set share=True to create a public link

# Main execution
if __name__ == "__main__":
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not all([groq_api_key, google_api_key, openai_api_key]):
            raise ValueError("One or more API keys are missing.")

        translator = TranslatorApp(groq_api_key, google_api_key, openai_api_key)
        gradio_app = GradioInterface(translator)

        gradio_app.launch()
    except Exception as e:
        print(f"Error: {e}")

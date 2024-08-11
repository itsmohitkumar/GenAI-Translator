import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from langdetect import detect, DetectorFactory

# Initialize language detection seed
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

class Logger:
    def __init__(self, log_dir="logs", log_file="translator_app.log"):
        # Ensure log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Logger configuration
        log_path = os.path.join(log_dir, log_file)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                RotatingFileHandler(log_path, maxBytes=10**6, backupCount=5)
            ]
        )
        self.logger = logging.getLogger("TranslatorApp")

    def get_logger(self):
        return self.logger

class APIClient:
    def __init__(self, api_key, model_name, client_class, logger):
        self.api_key = api_key
        self.model_name = model_name
        self.client_class = client_class
        self.logger = logger

    def create_client(self):
        try:
            return self.client_class(
                api_key=self.api_key,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                model=self.model_name
            )
        except TypeError as e:
            self.logger.error(f"Error initializing client: {e}")
            raise

class Config:
    def __init__(self, logger):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.logger = logger

        if not all([self.groq_api_key, self.google_api_key, self.openai_api_key]):
            self.logger.error("One or more API keys are missing in the environment variables.")
            raise ValueError("One or more API keys are missing in the environment variables.")

class TranslatorApp:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.groq_chain = self._create_chain(APIClient(self.config.groq_api_key, "llama3-70b-8192", ChatGroq, self.logger))
        self.google_chain = self._create_chain(APIClient(self.config.google_api_key, "gemini-1.5-pro", ChatGoogleGenerativeAI, self.logger))
        self.openai_chain = self._create_chain(APIClient(self.config.openai_api_key, "gpt-4o", ChatOpenAI, self.logger))

    def _create_chain(self, api_client):
        try:
            chatbot = api_client.create_client()
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an advanced translation assistant equipped with powerful language models. Your task is to accurately translate the provided text from {input_language} to {output_language}. 

                **Instructions:**

                1. **Translation Output:**
                - Provide a precise and fluent translation of the text. Ensure the translation maintains the original meaning and context.

                2. **Suggestions and Insights:**
                - Offer additional context or suggestions related to the translation, if applicable. This can include:
                    - Explanations of cultural nuances.
                    - Alternative translations or phrasing.
                    - Notes on any ambiguities or challenges in the translation.

                **Guidelines:**
                - Make sure the translation is clear, contextually accurate, and grammatically correct.
                - Ensure that suggestions are relevant and enhance the quality of the translation.

                **Input Text:**
                {input}
                    """),
                ("human", "{input}")
            ])
            return prompt | chatbot | StrOutputParser()
        except Exception as e:
            self.logger.error(f"Error creating translation chain: {e}")
            raise

    def _translate(self, chain, input_language, output_language, input_text):
        try:
            if chain is None:
                return {"translation": "Translation chain not available.", "insights": ""}
            result = chain.invoke({
                "input_language": input_language,
                "output_language": output_language,
                "input": input_text,
            })
            # Assuming the result contains the translation and insights separated by a delimiter, e.g., "\n\nSuggestions and Insights:"
            translation, insights = result.split("\n\n**Suggestions and Insights:**", 1) if "\n\nSuggestions and Insights:" in result else (result, "")
            return {"translation": translation.strip(), "insights": insights.strip()}
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            return {"translation": "An unexpected error occurred during translation.", "insights": ""}

    def perform_translations(self, model_name, output_language, input_text):
        chain_map = {
            "Groq": self.groq_chain,
            "Google": self.google_chain,
            "OpenAI": self.openai_chain,
        }
        selected_chain = chain_map.get(model_name)
        if selected_chain:
            try:
                input_language = detect(input_text)
                return self._translate(selected_chain, input_language, output_language, input_text)
            except Exception as e:
                self.logger.error(f"Error detecting language or performing translation: {e}")
                return {"translation": "Error detecting language or performing translation.", "insights": ""}
        return {"translation": "Invalid model selected.", "insights": ""}

class GradioInterface:
    def __init__(self, translator_app):
        self.translator = translator_app
        self._create_interface()

    def _create_interface(self):
        def translate_text(model_name, input_text, output_language):
            result = self.translator.perform_translations(model_name, output_language, input_text)
            return result["translation"]

        def copy_translation(output_text):
            return f"Copied! {output_text}"

        with gr.Blocks(css=".gradio-container { max-width: 800px; margin: auto; }") as demo:
            gr.Markdown("# 📝 Multilingual Translator")
            gr.Markdown("Translate text between different languages using your chosen chatbot model.")

            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        choices=["Groq", "Google", "OpenAI"],
                        label="Choose Translation Model",
                        value="Groq"
                    )
                    input_text = gr.Textbox(label="Input Text", lines=5, placeholder="Enter text here...")
                    output_language = gr.Dropdown(
                        choices=["English", "Hindi", "German", "French", "Spanish", "Chinese", "Japanese"],
                        label="Output Language",
                        value="German"
                    )
                    translate_button = gr.Button("Translate")

                with gr.Column(scale=1):
                    output_text = gr.Textbox(label="Translation Output", lines=5, placeholder="Translation will appear here...")
                    copy_button = gr.Button("Copy to Clipboard")
                    copied_message = gr.Markdown("")

                    clear_button = gr.Button("Clear")

            translate_button.click(translate_text, inputs=[model_choice, input_text, output_language], outputs=output_text)
            clear_button.click(lambda: ("", ""), None, [output_text, copied_message])

            copy_button.click(copy_translation, inputs=output_text, outputs=copied_message)
        
        self.interface = demo

    def launch(self):
        self.interface.launch(share=True)

if __name__ == "__main__":
    try:
        # Initialize Logger
        logger_instance = Logger()
        logger = logger_instance.get_logger()

        # Initialize Config
        config = Config(logger)

        # Initialize Translator Application
        translator = TranslatorApp(config)

        # Launch Gradio Interface
        gradio_app = GradioInterface(translator)
        gradio_app.launch()
    except Exception as e:
        logger.error(f"Error: {e}")

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
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Initialize language detection seed
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

class Logger:
    def __init__(self, log_dir, log_file):
        """
        Initialize the Logger with a directory and file name.
        Sets up logging to both console and file with rotation.
        """
        os.makedirs(log_dir, exist_ok=True)
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
        """
        Return the logger instance.
        """
        return self.logger

class Config:
    def __init__(self, logger):
        """
        Load configuration from environment variables and initialize settings.
        """
        self.logger = logger
        self._load_environment_variables()
        self._set_default_values()
        self._validate_config()

    def _load_environment_variables(self):
        """
        Load API keys from environment variables.
        """
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def _set_default_values(self):
        """
        Set default values for model names and other settings.
        """
        self.groq_model_name = "llama3-70b-8192"
        self.google_model_name = "gemini-1.5-pro"
        self.openai_model_name = "gpt-4o"
        self.temperature = 0
        self.max_tokens = None
        self.timeout = None
        self.max_retries = 2
        self.log_dir = "logs"
        self.log_file = "translator_app.log"
        self.default_input_language = "English"
        self.default_output_language = "German"
        self.gradio_css = ".gradio-container { max-width: 800px; margin: auto; }"

    def _validate_config(self):
        """
        Validate the loaded configuration.
        """
        missing_keys = []
        if not self.groq_api_key:
            missing_keys.append("GROQ_API_KEY")
        if not self.google_api_key:
            missing_keys.append("GOOGLE_API_KEY")
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")

        if missing_keys:
            self.logger.error(f"Missing environment variables: {', '.join(missing_keys)}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")

class APIClient:
    def __init__(self, api_key, model_name, client_class, config):
        """
        Initialize APIClient with API key, model name, client class, and config.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client_class = client_class
        self.config = config
        self.client = None

    def create_client(self):
        """
        Create and return the client instance.
        """
        if self.client is None:
            try:
                self.client = self.client_class(
                    api_key=self.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                    model=self.model_name
                )
            except TypeError as e:
                self.config.logger.error(f"Error initializing client: {e}")
                raise
        return self.client

class TranslatorApp:
    def __init__(self, config):
        """
        Initialize the Translator Application with the given config.
        Sets up translation chains for different models.
        """
        self.config = config
        self.logger = config.logger
        self._initialize_clients()
        self._create_chains()

    def _initialize_clients(self):
        """
        Initialize API clients for different models.
        """
        self.groq_client = APIClient(self.config.groq_api_key, self.config.groq_model_name, ChatGroq, self.config)
        self.google_client = APIClient(self.config.google_api_key, self.config.google_model_name, ChatGoogleGenerativeAI, self.config)
        self.openai_client = APIClient(self.config.openai_api_key, self.config.openai_model_name, ChatOpenAI, self.config)

    def _create_chains(self):
        """
        Create translation chains for different models.
        """
        self.groq_chain = self._create_chain(self.groq_client)
        self.google_chain = self._create_chain(self.google_client)
        self.openai_chain = self._create_chain(self.openai_client)

    def _create_chain(self, api_client):
        """
        Create a chain for translation using the specified API client.
        """
        try:
            chatbot = api_client.create_client()
            prompt = ChatPromptTemplate.from_messages([
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
            return prompt | chatbot | StrOutputParser()
        except Exception as e:
            self.logger.error(f"Error creating translation chain: {e}")
            raise

    def _translate(self, chain, input_language, output_language, input_text):
        """
        Perform translation using the specified chain.
        """
        try:
            if chain is None:
                return {"translation": "Translation chain not available.", "insights": ""}
            result = chain.invoke({
                "input_language": input_language,
                "output_language": output_language,
                "input": input_text,
            })
            translation, insights = result.split("\n\n**Suggestions and Insights:**", 1) if "\n\n**Suggestions and Insights:**" in result else (result, "")
            return {"translation": translation.strip(), "insights": insights.strip()}
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            return {"translation": "An unexpected error occurred during translation.", "insights": ""}

    def perform_translations(self, model_name, output_language, input_text):
        """
        Perform translations using the specified model.
        """
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
        """
        Initialize the Gradio interface for the Translator Application.
        """
        self.translator = translator_app
        self._create_interface()

    def _create_interface(self):
        """
        Create the Gradio interface for the application.
        """
        def translate_text(model_name, input_text, output_language):
            """
            Translate text based on the chosen model and parameters.
            """
            result = self.translator.perform_translations(model_name, output_language, input_text)
            return result["translation"]

        def copy_translation(output_text):
            """
            Return a message indicating the translation was copied.
            """
            return f"Copied! {output_text}"

        def clear_fields():
            """
            Clear the output text and copied message fields.
            """
            return "", ""

        with gr.Blocks(css=self.translator.config.gradio_css) as demo:
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
                        choices=[
                                "English", "Hindi", "German", "French", "Spanish", "Chinese", "Japanese", 
                                "Korean", "Russian", "Arabic", "Portuguese", "Italian", "Turkish", 
                                "Dutch", "Swedish", "Norwegian", "Danish", "Polish", "Czech", 
                                "Greek", "Hungarian"
                            ],
                        label="Output Language",
                        value=self.translator.config.default_output_language
                    )
                    translate_button = gr.Button("Translate")

                with gr.Column(scale=1):
                    output_text = gr.Textbox(label="Translation Output", lines=5, placeholder="Translation will appear here...")
                    copy_button = gr.Button("Copy to Clipboard")
                    copied_message = gr.Markdown("")
                    clear_button = gr.Button("Clear")

            translate_button.click(translate_text, inputs=[model_choice, input_text, output_language], outputs=output_text)
            clear_button.click(clear_fields, None, [output_text, copied_message])
            copy_button.click(copy_translation, inputs=output_text, outputs=copied_message)

        self.interface = demo

    def launch(self):
        """
        Launch the Gradio interface.
        """
        self.interface.launch(share=True)

if __name__ == "__main__":
    try:
        # Initialize Logger
        logger_instance = Logger(log_dir="logs", log_file="translator_app.log")
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

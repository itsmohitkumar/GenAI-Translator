import os
from dotenv import load_dotenv
from src.logger import Logger

class Config:
    def __init__(self):
        """
        Load configuration from environment variables and initialize settings.
        """
        self.logger = self._initialize_logger()
        self._load_environment_variables()
        self.setup_langchain()
        self._set_default_values()
        self._validate_config()

    def _initialize_logger(self):
        """
        Initialize and return a logger instance.
        """
        logger_instance = Logger(log_dir="logs", log_file="translator_app.log")
        return logger_instance.get_logger()

    def _load_environment_variables(self):
        """
        Load API keys from environment variables.
        """
        load_dotenv()  # Ensure environment variables are loaded
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def setup_langchain(self):
        """
        Setup LangChain environment variables and project.
        """
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        if not langchain_api_key:
            raise ValueError("LANGCHAIN_API_KEY environment variable is not set.")
        
        # Set project name and environment variables
        project_name = "translator_app"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

    def _set_default_values(self):
        """
        Set default values for model names and other settings.
        """
        self.groq_model_name = "llama3-70b-8192"
        self.google_model_name = "gemini-1.5-pro"
        self.openai_model_name = "gpt-3.5-turbo"
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

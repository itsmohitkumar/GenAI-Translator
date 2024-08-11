from dotenv import load_dotenv
from src.translator_app.config import Config
from src.translator_app.translator import TranslatorApp
from src.translator_app.gradio_interface import GradioInterface

def main():
    try:
        # Load environment variables
        load_dotenv()

        # Initialize Config (which initializes Logger)
        config = Config()

        # Initialize Translator Application
        translator = TranslatorApp(config)

        # Launch Gradio Interface
        gradio_app = GradioInterface(translator)
        gradio_app.launch()
    except Exception as e:
        # Use the logger from Config
        config.logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()

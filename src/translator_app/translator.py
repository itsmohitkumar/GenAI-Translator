from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect
from src.translator_app.api_client import APIClient

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
            # Split result to separate translation and insights
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

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

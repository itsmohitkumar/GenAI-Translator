# GenAI Translator

**GenAI Translator** is a state-of-the-art multilingual translation application designed to leverage advanced language models for high-quality translations. This project uses powerful AI models from Groq, Google, and OpenAI to deliver precise and contextually accurate translations.

![YouTube Echo Demo](image/translater-demo.png)

## Features

- **Multi-Model Support**: Integrates with Groq, Google, and OpenAI language models.
- **Customizable Prompts**: Uses configurable prompt templates for translation tasks.
- **User-Friendly Interface**: Built with Gradio for easy interaction and visualization.
- **Logging & Monitoring**: Comprehensive logging setup with file rotation for tracking and debugging.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [File Structure](#file-structure)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/itsmohitkumar/GenAI-Translator.git
    cd GenAI-Translator
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**

    Create a `.env` file in the root directory and add your API keys:

    ```env
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
    OPENAI_API_KEY=your_openai_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key
    ```

## Configuration

The configuration is managed in the `config.py` file. The default values can be adjusted based on your preferences.

### Default Values:

- **Groq Model Name**: `llama3-70b-8192`
- **Google Model Name**: `gemini-1.5-pro`
- **OpenAI Model Name**: `gpt-3.5-turbo`
- **Temperature**: `0`
- **Max Tokens**: `None`
- **Timeout**: `None`
- **Max Retries**: `2`
- **Default Input Language**: `English`
- **Default Output Language**: `German`

## Usage

1. **Run the Application:**

    ```bash
    python app.py
    ```

2. **Access the Gradio Interface:**

    Open the link provided in the terminal output to access the translation interface.

3. **Using the TranslatorApp Package:**

    You can use the `translator-app` package for programmatic access to the translation functionality. First, install the package:

    ```bash
    pip install translator-app==0.1
    ```

    Then, use the following code snippet to perform translations:

    ```python
    from translator_app.config import Config
    from translator_app.translator import TranslatorApp

    # Initialize configuration
    config = Config()

    # Create a TranslatorApp instance with the configuration
    translator = TranslatorApp(config)

    # Perform a translation
    result = translator.perform_translations("Groq", "German", "Hello World")

    # Print the result
    print(result)
    ```

## File Structure

```
GenAI-Translator/
│
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── prompts.py
│
├── translator_app/
│   ├── __init__.py
│   ├── config.py
│   ├── api_client.py
│   ├── translator.py
│   └── gradio_interface.py
│
├── app.py
├── .env
├── requirements.txt
└── README.md
```

- **`src/`**: Contains core modules including logging and prompt templates.
- **`translator_app/`**: Contains configuration, API client, translation logic, and Gradio interface.
- **`app.py`**: Entry point to run the application.
- **`.env`**: Environment variables configuration file.
- **`requirements.txt`**: List of dependencies for the project.
- **`README.md`**: Project documentation.

## API Documentation

- **APIClient**: Handles API communication for different language models.
- **TranslatorApp**: Manages translation logic and model interactions.
- **GradioInterface**: Provides a web interface for user interaction.

## Contributing

We welcome contributions to the GenAI Translator project! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

Please ensure your code adheres to the project's style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please contact:

- **Author**: Mohit Kumar
- **Email**: [mohitpanghal12345@gmail.com](mailto:mohitpanghal12345@gmail.com)

---

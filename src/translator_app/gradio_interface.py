import gradio as gr

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

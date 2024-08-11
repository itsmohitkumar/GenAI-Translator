import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/logger.py",
    "src/prompts.py",
    "src/translator_app/__init__.py",
    "src/translator_app/ai.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "test.py",
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "static/.gitkeep",
    "templates/chat.html"
]

for filepath in list_of_files:
    path = Path(filepath)
    filedir, filename = path.parent, path.name

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if not path.exists() or path.stat().st_size == 0:
        with open(path, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already created")

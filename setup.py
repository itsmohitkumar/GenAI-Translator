from setuptools import find_packages, setup

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='genai-translator',
    version='0.1',
    author='Mohit Kumar',
    author_email='mohitpanghal12345@gmail.com',
    description='An AI-powered chatbot that leverages multiple open-source models, including Groq, for advanced language processing and translation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/itsmohitkumar/GenAI-Translator',
    install_requires=[
        'langchain_groq',
        'langchain_google_genai',
        'langchain_openai',
        'gradio',
        'langdetect',
        'python-dotenv'
    ],
    packages=find_packages(include=['translator_app', 'src']),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'translator_app=app:main',
        ],
    },
)

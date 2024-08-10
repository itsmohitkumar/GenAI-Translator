from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='bharat-chatbot-groq',
    version='0.0.1',
    author='Mohit Kumar',
    author_email='mohitpanghal12345@gmail.com',
    description='An AI-powered chatbot using multiple open source models using groq',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/itsmohitkumar/bharat-chatbot-groq',
    install_requires=[
        'langchain',
        'python-dotenv',
        'langchain-core',
        'langchain-community',
        'langchain-groq',
        'sentence-transformers',
        'sentence_transformers'
    ],
    packages=find_packages(),
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
)
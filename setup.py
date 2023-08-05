from setuptools import setup, find_packages

setup(
    name='litellm',
    version='0.1.235',
    description='Library to easily interface with LLM API providers',
    author='BerriAI',
    packages=[
        'litellm'
    ],
    package_data={
        "litellm": ["integrations/*"],  # Specify the directory path relative to your package
    },
    install_requires=[
        'openai',
        'cohere',
        'pytest',
        'anthropic',
        'replicate',
        'python-dotenv',
        'openai[datalib]',
        'tenacity'
    ],
)

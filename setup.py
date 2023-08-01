from setuptools import setup, find_packages

setup(
    name='litellm',
    version='0.1.206',
    description='Library to easily interface with LLM API providers',
    author='BerriAI',
    packages=[
        'litellm'
    ],
    install_requires=[
        'openai',
        'cohere',
        'func_timeout',
        'pytest',
        'anthropic',
        'replicate',
        'python-dotenv',
        'openai[datalib]'
    ],
)

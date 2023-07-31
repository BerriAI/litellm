from setuptools import setup, find_packages

setup(
    name='litellm',
    version='0.1.2',
    description='Library to easily interface with LLM API providers',
    author='BerriAI',
    packages=[
        'litellm'
    ],
    install_requires=[
        'openai',
        'cohere'
    ],
)

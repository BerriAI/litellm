## LOCAL TEST
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
# from langchain.schema import HumanMessage, SystemMessage

# chat = ChatOpenAI(
#     openai_api_base="http://0.0.0.0:8000",
#     model = "gpt-3.5-turbo",
#     temperature=0.1
# )

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant that im using to make a test request to."
#     ),
#     HumanMessage(
#         content="test from litellm. tell me why it's amazing in 1 sentence"
#     ),
# ]
# response = chat(messages)

# print(response)

# claude_chat = ChatOpenAI(
#     openai_api_base="http://0.0.0.0:8000",
#     model = "claude-v1",
#     temperature=0.1
# )

# response = claude_chat(messages)

# print(response)

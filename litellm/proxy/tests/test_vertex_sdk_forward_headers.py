# import datetime

# import vertexai
# from vertexai.generative_models import Part
# from vertexai.preview import caching
# from vertexai.preview.generative_models import GenerativeModel

# LITE_LLM_ENDPOINT = "http://localhost:4000"

# vertexai.init(
#     project="adroit-crow-413218",
#     location="us-central1",
#     api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex-ai",
#     api_transport="rest",
# )

# # model = GenerativeModel(model_name="gemini-1.5-flash-001")
# # response = model.generate_content(
# #     "hi tell me a joke and a very long story", stream=True
# # )

# # print("response", response)

# # for chunk in response:
# #     print(chunk)


# system_instruction = """
# You are an expert researcher. You always stick to the facts in the sources provided, and never make up new facts.
# Now look at these research papers, and answer the following questions.
# """

# contents = [
#     Part.from_uri(
#         "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
#         mime_type="application/pdf",
#     ),
#     Part.from_uri(
#         "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
#         mime_type="application/pdf",
#     ),
# ]

# cached_content = caching.CachedContent.create(
#     model_name="gemini-1.5-pro-001",
#     system_instruction=system_instruction,
#     contents=contents,
#     ttl=datetime.timedelta(minutes=60),
#     # display_name="example-cache",
# )

# print(cached_content.name)

import requests


def get_next_url(response):
    """
    Function to get 'next' url from Link header
    :param response: response from requests
    :return: next url or None
    """
    if "link" not in response.headers:
        return None
    headers = response.headers

    next_url = headers["Link"]
    print(next_url)
    start_index = next_url.find("<")
    end_index = next_url.find(">")

    return next_url[1:end_index]


def get_models(url):
    """
    Function to retrieve all models from paginated endpoint
    :param url: base url to make GET request
    :return: list of all models
    """
    models = []
    while url:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return models
        payload = response.json()
        url = get_next_url(response)
        models.extend(payload)
    return models


def get_cleaned_models(models):
    """
    Function to clean retrieved models
    :param models: list of retrieved models
    :return: list of cleaned models
    """
    cleaned_models = []
    for model in models:
        cleaned_models.append(model["id"])
    return cleaned_models


# Get text-generation models
url = "https://huggingface.co/api/models?filter=text-generation-inference"
text_generation_models = get_models(url)
cleaned_text_generation_models = get_cleaned_models(text_generation_models)

print(cleaned_text_generation_models)


# Get conversational models
url = "https://huggingface.co/api/models?filter=conversational"
conversational_models = get_models(url)
cleaned_conversational_models = get_cleaned_models(conversational_models)

print(cleaned_conversational_models)


def write_to_txt(cleaned_models, filename):
    """
    Function to write the contents of a list to a text file
    :param cleaned_models: list of cleaned models
    :param filename: name of the text file
    """
    with open(filename, "w") as f:
        for item in cleaned_models:
            f.write("%s\n" % item)


# Write contents of cleaned_text_generation_models to text_generation_models.txt
write_to_txt(
    cleaned_text_generation_models,
    "huggingface_llms_metadata/hf_text_generation_models.txt",
)

# Write contents of cleaned_conversational_models to conversational_models.txt
write_to_txt(
    cleaned_conversational_models,
    "huggingface_llms_metadata/hf_conversational_models.txt",
)

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import litellm
import requests
from bs4 import BeautifulSoup

# URL of the AWS Bedrock Pricing page
PRICING_URL = "https://aws.amazon.com/bedrock/pricing/"

# List of providers to extract pricing for
PROVIDERS = ["ai21", "anthropic", "meta", "cohere", "mistral", "stability", "amazon"]


def extract_amazon_pricing(section):
    """
    Extracts pricing data for Amazon-specific models.

    Args:
        section (Tag): The BeautifulSoup Tag object for the Amazon section.

    Returns:
        dict: Pricing data for Amazon models.
    """
    tabs = section.find_all("li", class_="lb-tabs-trigger")
    panels = section.find_all("li", class_="lb-tabs-content-item")

    amazon_pricing = {}

    for tab, panel in zip(tabs, panels):
        model_name = tab.get_text(strip=True)
        table = panel.find("table")
        if not table:
            amazon_pricing[model_name] = "Pricing table not found"
            continue

        # Parse the table
        rows = table.find_all("tr")
        headers = [header.get_text(strip=True) for header in rows[0].find_all("td")]
        model_pricing = {}

        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue  # Skip rows with insufficient data

            feature_name = cols[0].get_text(strip=True)
            input_price = cols[1].get_text(strip=True)
            output_price = cols[2].get_text(strip=True)
            model_pricing[feature_name] = {
                headers[1]: input_price,
                headers[2]: output_price,
            }

        amazon_pricing[model_name] = model_pricing

    return amazon_pricing


def get_bedrock_pricing(url, providers):
    """
    Fetches and parses AWS Bedrock pricing for specified providers.

    Args:
        url (str): URL of the AWS Bedrock pricing page.
        providers (list): List of providers to extract pricing for.

    Returns:
        dict: A dictionary containing pricing data for the providers.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    pricing_data = {}

    for provider in providers:
        if provider == "amazon":
            section = soup.find(
                "li",
                class_="lb-tabs-accordion-trigger",
                text=lambda t: t and "Amazon" in t,
            )
            if not section:
                pricing_data[provider] = "Amazon section not found"
                continue

            amazon_section = section.find_next("li", class_="lb-tabs-content-item")
            if not amazon_section:
                pricing_data[provider] = "Amazon models section not found"
                continue

            pricing_data[provider] = extract_amazon_pricing(amazon_section)
        else:
            # General logic for other providers
            section = soup.find(
                "h2", text=lambda t: t and provider.lower() in t.lower()
            )
            if not section:
                pricing_data[provider] = "Provider section not found"
                continue

            table = section.find_next("table")
            if not table:
                pricing_data[provider] = "Pricing table not found"
                continue

            rows = table.find_all("tr")
            headers = [header.get_text(strip=True) for header in rows[0].find_all("td")]
            provider_pricing = {}

            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue

                model_name = cols[0].get_text(strip=True)
                input_price = cols[1].get_text(strip=True)
                output_price = cols[2].get_text(strip=True)
                provider_pricing[model_name] = {
                    "Price per 1,000 input tokens": input_price,
                    "Price per 1,000 output tokens": output_price,
                }

            pricing_data[provider] = provider_pricing

    return pricing_data


model_substring_map = {
    "ai21": {"jurassic-2": "j2"},
    "anthropic": {"claude-2-1": "claude-v2:1", "claude-2-0": "claude-v2"},
    "meta": {"llama-2-chat-(13b)": "llama2-13b-chat"},
    "cohere": {
        "r+": "r-plus",
        "embed-3-english": "embed-english-v3",
        "embed-3-multilingual": "embed-multilingual-v3",
    },
}  # aliases used by bedrock in their real model name vs. pricing page


def _handle_meta_model_name(model_name: str) -> str:
    # Check if it's a Llama 2 chat model
    if "llama-2-chat-" in model_name.lower():
        # Extract the size (e.g., 13b, 70b) using string manipulation
        # Look for pattern between "chat-(" and ")"
        import re

        if match := re.search(r"chat-\((\d+b)\)", model_name.lower()):
            size = match.group(1)
            return f"meta.llama2-{size}-chat"
    return model_name


def _handle_cohere_model_name(model_name: str) -> str:
    if model_name.endswith("command-r"):
        return "cohere.command-r-v1"
    return model_name


def _create_bedrock_model_name(provider: str, model_name: str):
    complete_model_name = f"{provider.lower()}.{model_name.replace(' ', '-').replace('.', '-').replace('*', '').lower()}"
    for provider_key, map in model_substring_map.items():
        if provider_key == provider:
            for model_substring, replacement in map.items():
                print(
                    f"model_substring: {model_substring}, replacement: {replacement}, received model_name: {model_name}"
                )
                if model_substring in complete_model_name:
                    print(f"model_name: {complete_model_name}")
                    complete_model_name = complete_model_name.replace(
                        model_substring, replacement
                    )
                    print(f"model_name: {complete_model_name}")
    if provider == "meta":
        complete_model_name = _handle_meta_model_name(complete_model_name)
    if provider == "cohere":
        complete_model_name = _handle_cohere_model_name(complete_model_name)
    return complete_model_name


def _convert_str_to_float(price_str: str) -> float:
    if "$" not in price_str:
        return 0.0
    return float(price_str.replace("$", ""))


def _check_if_model_name_in_pricing(
    bedrock_model_name: str,
    input_cost_per_1k_tokens: str,
    output_cost_per_1k_tokens: str,
):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    for model, value in litellm.model_cost.items():
        if model.startswith(bedrock_model_name):
            input_cost_per_token = (
                _convert_str_to_float(input_cost_per_1k_tokens) / 1000
            )
            output_cost_per_token = (
                _convert_str_to_float(output_cost_per_1k_tokens) / 1000
            )
            assert round(value["input_cost_per_token"], 10) == round(
                input_cost_per_token, 10
            ), f"Invalid input cost per token for {model} \n Bedrock pricing page name={bedrock_model_name} \n Got={value['input_cost_per_token']}, Expected={input_cost_per_token}"
            assert round(value["output_cost_per_token"], 10) == round(
                output_cost_per_token, 10
            ), f"Invalid output cost per token for {model} \n Bedrock pricing page name={bedrock_model_name} \n Got={value['output_cost_per_token']}, Expected={output_cost_per_token}"
            return True
    return False


if __name__ == "__main__":
    try:
        pricing = get_bedrock_pricing(PRICING_URL, PROVIDERS)
        print("AWS Bedrock On-Demand Pricing:")
        for provider, data in pricing.items():
            print(f"\n{provider.capitalize()}:")
            if isinstance(data, dict):
                for model, details in data.items():
                    complete_model_name = _create_bedrock_model_name(provider, model)
                    print(f"details: {details}")
                    assert _check_if_model_name_in_pricing(
                        bedrock_model_name=complete_model_name,
                        input_cost_per_1k_tokens=details[
                            "Price per 1,000 input tokens"
                        ],
                        output_cost_per_1k_tokens=details[
                            "Price per 1,000 output tokens"
                        ],
                    ), f"Model {complete_model_name} not found in litellm.model_cost"
                    print(f"  {complete_model_name}:")
                    if isinstance(details, dict):
                        for detail, value in details.items():
                            print(f"    {detail}: {value}")
                    else:
                        print(f"    {details}")
            else:
                print(f"  {data}")
    except requests.RequestException as e:
        print(f"Error fetching pricing data: {e}")

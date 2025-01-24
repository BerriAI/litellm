import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

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
                    headers[1]: input_price,
                    headers[2]: output_price,
                }

            pricing_data[provider] = provider_pricing

    return pricing_data


if __name__ == "__main__":
    try:
        pricing = get_bedrock_pricing(PRICING_URL, PROVIDERS)
        print("AWS Bedrock On-Demand Pricing:")
        for provider, data in pricing.items():
            print(f"\n{provider.capitalize()}:")
            if isinstance(data, dict):
                for model, details in data.items():
                    print(f"  {model}:")
                    if isinstance(details, dict):
                        for detail, value in details.items():
                            print(f"    {detail}: {value}")
                    else:
                        print(f"    {details}")
            else:
                print(f"  {data}")
    except requests.RequestException as e:
        print(f"Error fetching pricing data: {e}")

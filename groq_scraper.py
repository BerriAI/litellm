import subprocess
from bs4 import BeautifulSoup
import pandas as pd
import json


def _get_table_from_heading_text(heading_text: str, soup: BeautifulSoup):
    llm_heading = soup.find('h4', string=lambda text: text and heading_text in text)
    heading_container = llm_heading.parent.parent
    table_container = heading_container.find_next_sibling('div', class_='elementor-widget')
    
    return table_container.find('table')


def _convert_name(full_name: str):
    new_name = full_name.split("(")[0].strip().lower()
    new_name = new_name.replace(" ", "-")
    return new_name


def _convert_price(price: str):
    ppm = float(price.split("\n")[0].replace("$", "").strip())
    return format(ppm / 1_000_000, '.8f')


def _extract_col_names(table: BeautifulSoup):
    col_names_soup = table.find_all('th')
    return [col_name_soup.text.lower().strip() for col_name_soup in col_names_soup]


def _extract_table_data(
        table: BeautifulSoup,
        desired_col_names: list[tuple[str, str]],
        col_map: dict={}
    ):
    col_names = _extract_col_names(table)

    for i, col in enumerate(col_names):
        for col_desc, litellm_name in desired_col_names:
            if col_desc in col:
                col_map[litellm_name] = i

    if "name" not in col_map and "try_now_button" not in col_map:
        raise ValueError("Neither model-name nor try-now-button column found")
    
    rows = table.find_all('tr')[1:]

    model_map = {}
    for row in rows:
        row_values = row.find_all('td')

        if "try_now_button" in col_map:
            model_groq_link = row_values[col_map["try_now_button"]].find('a').get('href')
            model_name = model_groq_link.split("?model=")[-1]
        else:
            model_name = row_values[col_map["name"]].text.lower().strip()

        model_map[model_name] = {}
        
        for col_name, col_idx in col_map.items():
            if col_name == "try_now_button" or col_name == "name":
                continue
            text_ = row_values[col_idx].text.strip()
            if "cost" in col_name:
                model_map[model_name][col_name] = _convert_price(text_)
            elif "supports" in col_name:
                if "yes" not in text_.lower() and "no" not in text_.lower():
                    raise ValueError(f"Invalid value for {col_name}: {text_}")
                model_map[model_name][col_name] = text_ == "Yes"
            else:
                model_map[model_name][col_name] = text_

    return model_map 


def scrape_groq_pricing():
    curl_command = [
        'curl',
        'https://groq.com/pricing/',
        '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        '-H', 'Accept-Language: en-US,en;q=0.9',
        '-H', 'Connection: keep-alive',
        '-H', 'Upgrade-Insecure-Requests: 1',
        '-H', 'Cache-Control: max-age=0',
        '--compressed'
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    soup = BeautifulSoup(result.stdout, 'html.parser')

    # Get table for chat models:
    chat_models_table = _get_table_from_heading_text("Large Language Models (LLMs)", soup)

    desired_col_names = [
        ("input token price", "input_cost_per_token"),
        ("output token price", "output_cost_per_token"),
    ]

    return _extract_table_data(chat_models_table, desired_col_names, {"try_now_button": 4})


def scrape_groq_capabilities():
    curl_command = [
        'curl',
        'https://console.groq.com/docs/tool-use',
        '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        '-H', 'Accept-Language: en-US,en;q=0.9',
        '-H', 'Connection: keep-alive',
        '-H', 'Upgrade-Insecure-Requests: 1',
        '-H', 'Cache-Control: max-age=0',
        '--compressed'
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    soup = BeautifulSoup(result.stdout, 'html.parser')

    capabilities_table = soup.find('table')

    desired_col_names = [
        ("model id", "name"),
        ("tool use support?", "supports_function_calling"),
        ("json mode support?", "supports_response_schema"),
    ]

    return _extract_table_data(capabilities_table, desired_col_names)


if __name__ == "__main__":
    scrape_groq_pricing()
    print("-" * 100)
    out = scrape_groq_capabilities()
    print(out)

import subprocess
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys
import re
import json

sys.path.insert(
    0, os.path.abspath("../..")
)
from workflows.auto_update_price_and_context_window_file import load_local_data, sync_local_data_with_remote, write_to_file


def _get_table_from_heading_text(heading_text: str, soup: BeautifulSoup):
    llm_heading = soup.find('h4', string=lambda text: text and heading_text in text)
    heading_container = llm_heading.parent.parent
    table_container = heading_container.find_next_sibling('div', class_='elementor-widget')
    
    return table_container.find('table')


def _convert_name(full_name: str):
    new_name = full_name.split("(")[0].strip().lower()
    new_name = new_name.replace(" ", "-")
    return new_name


def _convert_date(date: str):
    date_parts = date.split("/")
    return f"20{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"


def _convert_price(price: str, divisor: int=1_000_000):
    ppm = float(price.split("\n")[0].replace("$", "").replace("*", "").strip())
    return ppm / divisor


def _extract_col_names(table: BeautifulSoup):
    col_names_soup = table.find_all('th')
    return [col_name_soup.text.lower().strip() for col_name_soup in col_names_soup]


def _extract_table_data(
        table: BeautifulSoup,
        desired_col_names: list[tuple[str, str]],
        col_map_input: dict={}
    ):
    col_map = col_map_input.copy()

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

            text_ = row_values[col_idx].text.lower().strip()
            if "cost" in col_name:
                if "per_token" in col_name:
                    model_map[model_name][col_name] = _convert_price(text_)
                elif "per_second" in col_name:
                    model_map[model_name][col_name] = _convert_price(text_, 60**2)
                elif "per_character" in col_name:
                    model_map[model_name][col_name] = _convert_price(text_)
            elif "supports" in col_name:
                if "yes" not in text_ and "no" not in text_:
                    raise ValueError(f"Invalid value for {col_name}: {text_}")
                model_map[model_name][col_name] = text_ == "yes"
            elif "tokens" in col_name:
                tokens = text_.replace(",", "").replace("k", "000")
                if tokens.isdigit():
                    model_map[model_name][col_name] = int(tokens)
            elif "date" in col_name:
                model_map[model_name][col_name] = _convert_date(text_)
            else:
                model_map[model_name][col_name] = text_

    return model_map 


def _insert_dict_in_raw_json(remote_data: dict, local_data: dict, local_data_raw_input: str):
    local_data_raw = str(local_data_raw_input)  # make a copy of input
    for update_model_id, updated_model_dict in remote_data.items():
        # Format dict into raw JSON 
        new_model_json_unindented = json.dumps(updated_model_dict, indent=4)
        new_model_json = ""
        for i, line in enumerate(new_model_json_unindented.splitlines()):
            if i not in [0, len(new_model_json_unindented.splitlines()) - 1]:
                new_model_json += " " * 4 + line + "\n"
            elif i == 0:
                new_model_json += line + "\n"
            else:
                new_model_json += " " * 4 + line

        # If the model already exists, update its values
        if update_model_id in local_data:
            local_data_raw = re.sub(
                r'"'+ str(update_model_id) + r'":\s*\{.*?\n\s*\},',
                f'"{update_model_id}": {new_model_json},',
                local_data_raw,
                flags=re.DOTALL
            )
        # If the model doesn't exist, add it to the file
        else:
            # Find the position to insert the new model (maintaining alphabetical order)
            model_keys = list(local_data.keys())
            model_keys.append(update_model_id)
            model_keys.sort()
            new_model_index = model_keys.index(update_model_id)
            
            # If it's the last model, append to the end
            if new_model_index == len(model_keys) - 1:
                local_data_raw += f'"{update_model_id}": {new_model_json},\n'
            else:
                # Find the model that should come after our new model
                next_model = model_keys[new_model_index + 1]
                # Insert before the next model
                local_data_raw = re.sub(
                    r'"'+ re.escape(next_model) + r'":\s*\{',
                    f'"{update_model_id}": {new_model_json},\n    "{next_model}": {{',
                    local_data_raw,
                    flags=re.DOTALL
                )

    return local_data_raw


def _update_deprecated_models_in_raw_json(deprecation_data: dict, local_data: dict, local_data_raw_input: str):
    local_data_raw = str(local_data_raw_input)  # make a copy of input
    counter = 0
    for name, date_dict in deprecation_data.items():
        if name not in local_data:
            continue

        # Format dict into raw JSON 
        updated_model_dict = local_data[name].copy()
        updated_model_dict.update(date_dict)

        new_model_json_unindented = json.dumps(updated_model_dict, indent=4)
        new_model_json = ""
        for i, line in enumerate(new_model_json_unindented.splitlines()):
            if i not in [0, len(new_model_json_unindented.splitlines()) - 1]:
                new_model_json += " " * 4 + line + "\n"
            elif i == 0:
                new_model_json += line + "\n"
            else:
                new_model_json += " " * 4 + line

        # If the model already exists, update its values
        if name in local_data:
            counter += 1
            local_data_raw = re.sub(
                r'"'+ str(name) + r'":\s*\{.*?\n\s*\},',
                f'"{name}": {new_model_json},',
                local_data_raw,
                flags=re.DOTALL
            )

    print(f"Added deprecation dates for {counter} models on Groq")
    return local_data_raw


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

    # Chat models
    chat_models_table = _get_table_from_heading_text("Large Language Models (LLMs)", soup)
    desired_col_names = [
        ("input token price", "input_cost_per_token"),
        ("output token price", "output_cost_per_token"),
    ]
    out = _extract_table_data(chat_models_table, desired_col_names, {"try_now_button": 4})

    # TTS models
    chat_models_table = _get_table_from_heading_text("Text-to-Speech", soup)
    desired_col_names = [
        ("price", "input_cost_per_character"),
    ]
    out |= _extract_table_data(chat_models_table, desired_col_names, {"try_now_button": 3})

    # ASR models
    chat_models_table = _get_table_from_heading_text("Automatic Speech Recognition (ASR) Models", soup)
    desired_col_names = [
        ("price", "input_cost_per_second"),
    ]
    out |= _extract_table_data(chat_models_table, desired_col_names, {"try_now_button": 3})

    return out


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
        ("tool use support?", "supports_tool_choice"),
        ("tool use support?", "supports_function_calling"),
        ("json mode support?", "supports_response_schema"),
    ]

    return _extract_table_data(capabilities_table, desired_col_names)


def scrape_groq_context_window():
    curl_command = [
        'curl',
        'https://console.groq.com/docs/models',
        '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    soup = BeautifulSoup(result.stdout, 'html.parser')

    out = {}
    for context_window_table in soup.find_all('table'):
        desired_col_names = [
            ("model id", "name"),
            ("context window", "max_input_tokens"),
            ("completion tokens", "max_output_tokens"),
            ("completion tokens", "max_tokens"),
        ]

        context_window_dict = _extract_table_data(context_window_table, desired_col_names)
        for model_name, model_dict in context_window_dict.items():
            if "max_input_tokens" in model_dict:
                if "max_output_tokens" not in model_dict or model_dict["max_output_tokens"] == "-":
                    model_dict["max_output_tokens"] = model_dict["max_input_tokens"]
                if "max_tokens" not in model_dict or model_dict["max_tokens"] == "-":
                    model_dict["max_tokens"] = model_dict["max_output_tokens"]
        
        out |= context_window_dict
    return out


def scrape_groq_reasoning_models():
    curl_command = [
        'curl',
        'https://console.groq.com/docs/reasoning',
        '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    soup = BeautifulSoup(result.stdout, 'html.parser')

    reasoning_models_table = soup.find('table')

    desired_col_names = [
        ("model id", "name")
    ]

    reasoning_models_dict = _extract_table_data(reasoning_models_table, desired_col_names)
    for k in reasoning_models_dict:
        reasoning_models_dict[k] = {
            "supports_reasoning": True,
        }

    return reasoning_models_dict


def scrape_groq_deprecated_models():
    curl_command = [
        'curl',
        'https://console.groq.com/docs/deprecations',
        '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    soup = BeautifulSoup(result.stdout, 'html.parser')

    total = {}
    for i, table in enumerate(soup.find_all('table')):
        desired_col_names = [
            ("shutdown date", "deprecation_date"),
        ]

        total |= _extract_table_data(table, desired_col_names, {"name": 0})
    
    out = {}
    for k, v in total.items():
        out[f"groq/{k}"] = v

    return out


def scrape_groq_main():
    pricing = scrape_groq_pricing()
    capabilities = scrape_groq_capabilities()
    context_window = scrape_groq_context_window()
    reasoning_models = scrape_groq_reasoning_models()
    
    all_models = pricing.keys() | capabilities.keys() | context_window.keys() | reasoning_models.keys()
    total = {}
    for model_name in all_models:
        model = f"groq/{model_name}"
        total[model] = pricing.get(model_name, {}) | capabilities.get(model_name, {}) | context_window.get(model_name, {}) | reasoning_models.get(model_name, {})
        
        if "input_cost_per_token" in total[model]:
            total[model]["mode"] = "chat"
        elif "input_cost_per_second" in total[model]:
            total[model]["output_cost_per_second"] = 0.
            total[model]["mode"] = "audio_transcription"
        elif "tts" in model:
            total[model]["mode"] = "audio_speech"
        else:
            total.pop(model)
            continue
        
        total[model]["litellm_provider"] = "groq"

        # Keep the original data but ensure it's in the correct order
        total[model] = {
            "max_tokens": total[model].get("max_tokens", None),
            "max_input_tokens": total[model].get("max_input_tokens", None),
            "max_output_tokens": total[model].get("max_output_tokens", None),
            "input_cost_per_token": total[model].get("input_cost_per_token", None),
            "output_cost_per_token": total[model].get("output_cost_per_token", None),
            "input_cost_per_second": total[model].get("input_cost_per_second", None),
            "output_cost_per_second": total[model].get("output_cost_per_second", None),
            "litellm_provider": total[model].get("litellm_provider", None),
            "mode": total[model].get("mode", None),
            "supports_function_calling": total[model].get("supports_function_calling", None),
            "supports_response_schema": total[model].get("supports_response_schema", None),
            "supports_reasoning": total[model].get("supports_reasoning", None),
            "supports_tool_choice": total[model].get("supports_tool_choice", None),
        }
        total[model] = {k: v for k, v in total[model].items() if v is not None}

    return total


def main():
    local_file_path = "../model_prices_and_context_window.json" 

    local_data_raw = load_local_data(local_file_path, raw=True)
    local_data = load_local_data(local_file_path)

    remote_data = scrape_groq_main()
    deprecated_model_data = scrape_groq_deprecated_models()

    if local_data and remote_data:
        # local_data_raw = _insert_dict_in_raw_json(remote_data, local_data, local_data_raw)
        local_data_raw = _update_deprecated_models_in_raw_json(deprecated_model_data, local_data, local_data_raw)
        write_to_file(local_file_path, local_data_raw, write_raw=True)
    else:
        print("Failed to fetch model data from either local file or URL.")


if __name__ == "__main__":
    main()

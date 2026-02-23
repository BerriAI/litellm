import json

# List of models to update
models_to_update = [
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002-v2",
    "ft:gpt-4o-2024-08-06",
    "ft:gpt-4o-mini-2024-07-18",
    "ft:gpt-3.5-turbo",
    "ft:davinci-002",
    "ft:babbage-002",
]


def update_model_prices(file_path):
    # Read the JSON file as text first to preserve number formatting
    with open(file_path, "r") as file:
        original_text = file.read()
        data = json.loads(original_text)

    # Update specified models
    for model_name in models_to_update:
        print("finding model", model_name)
        if model_name in data:
            print("found model")
            model = data[model_name]
            if "input_cost_per_token" in model:
                # Format new values to match original style
                model["input_cost_per_token_batches"] = float(
                    "{:.12f}".format(model["input_cost_per_token"] / 2)
                )
            if "output_cost_per_token" in model:
                model["output_cost_per_token_batches"] = float(
                    "{:.12f}".format(model["output_cost_per_token"] / 2)
                )
        print("new pricing for model=")
        # Convert all float values to full decimal format before printing
        formatted_model = {
            k: "{:.9f}".format(v) if isinstance(v, float) else v
            for k, v in data[model_name].items()
        }
        print(json.dumps(formatted_model, indent=4))


# Run the update
file_path = "model_prices_and_context_window.json"
update_model_prices(file_path)

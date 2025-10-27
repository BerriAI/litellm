from litellm import completion, completion_cost
import time
import click
from tqdm import tqdm
from tabulate import tabulate
from termcolor import colored
import os


# Define the list of models to benchmark
# select any LLM listed here: https://docs.litellm.ai/docs/providers
models = ["gpt-3.5-turbo", "claude-2"]

# Enter LLM API keys
# https://docs.litellm.ai/docs/providers
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

# List of questions to benchmark (replace with your questions)
questions = ["When will BerriAI IPO?", "When will LiteLLM hit $100M ARR?"]

# Enter your system prompt here
system_prompt = """
You are LiteLLMs helpful assistant
"""


@click.command()
@click.option(
    "--system-prompt",
    default="You are a helpful assistant that can answer questions.",
    help="System prompt for the conversation.",
)
def main(system_prompt):
    for question in questions:
        data = []  # Data for the current question

        with tqdm(total=len(models)) as pbar:
            for model in models:
                colored_description = colored(
                    f"Running question: {question} for model: {model}", "green"
                )
                pbar.set_description(colored_description)
                start_time = time.time()

                response = completion(
                    model=model,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                )

                end = time.time()
                total_time = end - start_time
                cost = completion_cost(completion_response=response)
                raw_response = response["choices"][0]["message"]["content"]

                data.append(
                    {
                        "Model": colored(model, "light_blue"),
                        "Response": raw_response,  # Colorize the response
                        "ResponseTime": colored(f"{total_time:.2f} seconds", "red"),
                        "Cost": colored(f"${cost:.6f}", "green"),  # Colorize the cost
                    }
                )

                pbar.update(1)

        # Separate headers from the data
        headers = ["Model", "Response", "Response Time (seconds)", "Cost ($)"]
        colwidths = [15, 80, 15, 10]

        # Create a nicely formatted table for the current question
        table = tabulate(
            [list(d.values()) for d in data],
            headers,
            tablefmt="grid",
            maxcolwidths=colwidths,
        )

        # Print the table for the current question
        colored_question = colored(question, "green")
        click.echo(f"\nBenchmark Results for '{colored_question}':")
        click.echo(table)  # Display the formatted table


if __name__ == "__main__":
    main()

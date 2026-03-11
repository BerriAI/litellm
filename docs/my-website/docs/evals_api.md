# /evals

LiteLLM Proxy supports OpenAI's Evaluations (Evals) API, allowing you to create, manage, and run evaluations to measure model performance against defined testing criteria.

## What are Evals?

OpenAI Evals API provides a structured way to:
- **Create Evaluations**: Define testing criteria and data sources for evaluating model outputs
- **Run Evaluations**: Execute evaluations against specific models and datasets
- **Track Results**: Monitor evaluation progress and review detailed results

## Quick Start

### Setup LiteLLM Proxy

First, start your LiteLLM Proxy server:

```bash
litellm --config config.yaml

# Proxy will run on http://localhost:4000
```

### Initialize OpenAI Client

```python
from openai import OpenAI

# Point to your LiteLLM Proxy
client = OpenAI(
    api_key="sk-1234",  # Your LiteLLM proxy API key
    base_url="http://localhost:4000"  # Your proxy URL
)
```


For async operations:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)
```

---

## Evaluation Management

### Create an Evaluation

Create an evaluation with testing criteria and data source configuration.

#### Example: Sentiment Classification Eval

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Create evaluation with label model grader
eval_obj = client.evals.create(
    name="Sentiment Classification",
    data_source_config={
        "type": "stored_completions",
        "metadata": {"usecase": "chatbot"}
    },
    testing_criteria=[
        {
            "type": "label_model",
            "model": "gpt-4o-mini",
            "input": [
                {
                    "role": "developer",
                    "content": "Classify the sentiment of the following statement as one of 'positive', 'neutral', or 'negative'"
                },
                {
                    "role": "user",
                    "content": "Statement: {{item.input}}"
                }
            ],
            "passing_labels": ["positive"],
            "labels": ["positive", "neutral", "negative"],
            "name": "Sentiment Grader"
        }
    ]
)

# Note: If you want to use model-specific credentials for this evaluation, you can specify the model name in the extra body parameters.

print(f"Created eval: {eval_obj.id}")
print(f"Eval name: {eval_obj.name}")
```

#### Example: Push Notifications Summarizer Monitoring

This example shows how to monitor prompt changes for regressions in a push notifications summarizer:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Define data source for stored completions
data_source_config = {
    "type": "stored_completions",
    "metadata": {
        "usecase": "push_notifications_summarizer"
    }
}

# Define grader criteria
GRADER_DEVELOPER_PROMPT = """
Label the following push notification summary as either correct or incorrect.
The push notification and the summary will be provided below.
A good push notification summary is concise and snappy.
If it is good, then label it as correct, if not, then incorrect.
"""

GRADER_TEMPLATE_PROMPT = """
Push notifications: {{item.input}}
Summary: {{sample.output_text}}
"""

push_notification_grader = {
    "name": "Push Notification Summary Grader",
    "type": "label_model",
    "model": "gpt-4o-mini",
    "input": [
        {
            "role": "developer",
            "content": GRADER_DEVELOPER_PROMPT,
        },
        {
            "role": "user",
            "content": GRADER_TEMPLATE_PROMPT,
        },
    ],
    "passing_labels": ["correct"],
    "labels": ["correct", "incorrect"],
}

# Create the evaluation
eval_result = await client.evals.create(
    name="Push Notification Completion Monitoring",
    metadata={"description": "This eval monitors completions"},
    data_source_config=data_source_config,
    testing_criteria=[push_notification_grader],
)

eval_id = eval_result.id
print(f"Created eval: {eval_id}")
```

### List Evaluations

Retrieve a list of all your evaluations with pagination support.

```python
# List all evaluations
evals_response = client.evals.list(
    limit=20,
    order="desc"
)

for eval in evals_response.data:
    print(f"Eval ID: {eval.id}, Name: {eval.name}")

# Check if there are more evals
if evals_response.has_more:
    # Fetch next page
    next_evals = client.evals.list(
        after=evals_response.last_id,
        limit=20
    )
```

### Get a Specific Evaluation

Retrieve details of a specific evaluation by ID.

```python
eval = client.evals.retrieve(
    eval_id="eval_abc123"
)

print(f"Eval ID: {eval.id}")
print(f"Name: {eval.name}")
print(f"Data Source: {eval.data_source_config}")
print(f"Testing Criteria: {eval.testing_criteria}")
```

### Update an Evaluation

Update evaluation metadata or name.

```python
updated_eval = client.evals.update(
    eval_id="eval_abc123",
    name="Updated Evaluation Name",
    metadata={
        "version": "2.0",
        "updated_by": "user@example.com"
    }
)

print(f"Updated eval: {updated_eval.name}")
```

### Delete an Evaluation

Permanently delete an evaluation.

```python
delete_response = client.evals.delete(
    eval_id="eval_abc123"
)

print(f"Deleted: {delete_response.deleted}")  # True
```

---

## Evaluation Runs

### Create a Run

Execute an evaluation by creating a run. The run processes your data through the model and applies testing criteria.

#### Using Stored Completions

First, generate some test data by making chat completions with metadata:

```python
from openai import AsyncOpenAI
import asyncio

client = AsyncOpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Generate test data with different prompt versions
push_notification_data = [
    """
- New message from Sarah: "Can you call me later?"
- Your package has been delivered!
- Flash sale: 20% off electronics for the next 2 hours!
""",
    """
- Weather alert: Thunderstorm expected in your area.
- Reminder: Doctor's appointment at 3 PM.
- John liked your photo on Instagram.
"""
]

PROMPTS = [
    (
        """
        You are a helpful assistant that summarizes push notifications.
        You are given a list of push notifications and you need to collapse them into a single one.
        Output only the final summary, nothing else.
        """,
        "v1"
    ),
    (
        """
        You are a helpful assistant that summarizes push notifications.
        You are given a list of push notifications and you need to collapse them into a single one.
        The summary should be longer than it needs to be and include more information than is necessary.
        Output only the final summary, nothing else.
        """,
        "v2"
    )
]

# Create completions with metadata for tracking
tasks = []
for notifications in push_notification_data:
    for (prompt, version) in PROMPTS:
        tasks.append(client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": notifications},
            ],
            metadata={
                "prompt_version": version,
                "usecase": "push_notifications_summarizer"
            }
        ))

await asyncio.gather(*tasks)
```

Now create runs to evaluate different prompt versions:

```python
# Grade prompt_version=v1
eval_run_result = await client.evals.runs.create(
    eval_id=eval_id,
    name="v1-run",
    data_source={
        "type": "completions",
        "source": {
            "type": "stored_completions",
            "metadata": {
                "prompt_version": "v1",
            }
        }
    }
)

print(f"Run ID: {eval_run_result.id}")
print(f"Status: {eval_run_result.status}")
print(f"Report URL: {eval_run_result.report_url}")

# Grade prompt_version=v2
eval_run_result_v2 = await client.evals.runs.create(
    eval_id=eval_id,
    name="v2-run",
    data_source={
        "type": "completions",
        "source": {
            "type": "stored_completions",
            "metadata": {
                "prompt_version": "v2",
            }
        }
    }
)

print(f"Run ID: {eval_run_result_v2.id}")
print(f"Report URL: {eval_run_result_v2.report_url}")
```

#### Using Completions with Different Models

Test how different models perform on the same inputs:

```python
# Test with GPT-4o using stored completions as input
tasks = []
for prompt_version in ["v1", "v2"]:
    tasks.append(client.evals.runs.create(
        eval_id=eval_id,
        name=f"gpt-4o-run-{prompt_version}",
        data_source={
            "type": "completions",
            "input_messages": {
                "type": "item_reference",
                "item_reference": "item.input",
            },
            "model": "gpt-4o",
            "source": {
                "type": "stored_completions",
                "metadata": {
                    "prompt_version": prompt_version,
                }
            }
        }
    ))

results = await asyncio.gather(*tasks)
for run in results:
    print(f"Report URL: {run.report_url}")
```

### List Runs

Get all runs for a specific evaluation.

```python
# List all runs for an evaluation
runs_response = client.evals.runs.list(
    eval_id="eval_abc123",
    limit=20,
    order="desc"
)

for run in runs_response.data:
    print(f"Run ID: {run.id}")
    print(f"Status: {run.status}")
    print(f"Name: {run.name}")
    if run.result_counts:
        print(f"Results: {run.result_counts.passed}/{run.result_counts.total} passed")
```

### Get Run Details

Retrieve detailed information about a specific run, including results.

```python
run = client.evals.runs.retrieve(
    eval_id="eval_abc123",
    run_id="run_def456"
)

print(f"Run ID: {run.id}")
print(f"Status: {run.status}")
print(f"Started: {run.started_at}")
print(f"Completed: {run.completed_at}")

# Check results
if run.result_counts:
    print(f"\nOverall Results:")
    print(f"Total: {run.result_counts.total}")
    print(f"Passed: {run.result_counts.passed}")
    print(f"Failed: {run.result_counts.failed}")
    print(f"Error: {run.result_counts.errored}")

# Per-criteria results
if run.per_testing_criteria_results:
    for criteria_result in run.per_testing_criteria_results:
        print(f"\nCriteria {criteria_result.testing_criteria_index}:")
        print(f"  Passed: {criteria_result.result_counts.passed}")
        print(f"  Average Score: {criteria_result.average_score}")
```

### Delete a Run

Permanently delete a run and its results.

```python
delete_response = await client.evals.runs.delete(
    eval_id="eval_abc123",
    run_id="run_def456"
)

print(f"Deleted: {delete_response.deleted}")  # True
print(f"Run ID: {delete_response.run_id}")
```


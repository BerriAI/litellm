# Custom Pricing - SageMaker, Azure, etc

Register custom pricing for sagemaker completion model. 

For cost per second pricing, you **just** need to register `input_cost_per_second`. 

```python
# !pip install boto3 
from litellm import completion, completion_cost 

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""


def test_completion_sagemaker():
    try:
        print("testing sagemaker")
        response = completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            input_cost_per_second=0.000420,
        )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        print(cost)
    except Exception as e:
        raise Exception(f"Error occurred: {e}")

```


## Cost Per Token (e.g. Azure)


```python
# !pip install boto3 
from litellm import completion, completion_cost 

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""


def test_completion_azure_model():
    try:
        print("testing azure custom pricing")
        # azure call
        response = completion(
          model = "azure/<your_deployment_name>", 
          messages = [{ "content": "Hello, how are you?","role": "user"}]
          input_cost_per_token=0.005,
          output_cost_per_token=1,
        )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        print(cost)
    except Exception as e:
        raise Exception(f"Error occurred: {e}")

test_completion_azure_model()
```
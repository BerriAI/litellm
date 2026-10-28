## File Structure 

### August 27th, 2024 

To make it easy to see how calls are transformed for each model/provider:

we are working on moving all supported litellm providers to a folder structure, where folder name is the supported litellm provider name. 

Each folder will contain a `*_transformation.py` file, which has all the request/response transformation logic, making it easy to see how calls are modified. 

E.g. `cohere/`, `bedrock/`. 
     
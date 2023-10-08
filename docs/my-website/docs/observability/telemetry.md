# Telemetry 

LiteLLM contains a telemetry feature that tells us what models are used, and what errors are hit.

## What is logged? 

Only the model name and exception raised is logged. 

## Why?
We use this information to help us understand how LiteLLM is used, and improve stability. 

## Opting out
If you prefer to opt out of telemetry, you can do this by setting `litellm.telemetry = False`. 
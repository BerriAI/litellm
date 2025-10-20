# Objective

Create a custom LiteLLM provider to add support for the new AWS AgentCore Runtime.

## Problem statement

AWS AgentCore Runtime is a new service from AWS for deploying agent framework agnostic agents (Strands, CrewAI, et.c.). LiteLLM currently does not support proxying requests to endpoints exposed by AgentCore, `/invoke`.

## IMPORTANT

- ALWAYS use the available `agentcore` mcp server that is running locally for up to date AgentCore documentation
- ALWAYS test your implementation by setting up an AgentCore Runtime in AWS and run LiteLLM locally with the new provider implementation

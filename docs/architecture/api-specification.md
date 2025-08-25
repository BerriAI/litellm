# API Specification

## REST API Specification

```yaml
openapi: 3.0.0
info:
  title: LiteLLM Proxy API
  version: 1.0.0
  description: OpenAI-compatible LLM gateway with multi-provider support
servers:
  - url: http://localhost:4000
    description: Local development server
  - url: https://api.litellm.ai
    description: Production server

paths:
  /chat/completions:
    post:
      summary: Create chat completion
      operationId: createChatCompletion
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatCompletionRequest'
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatCompletionResponse'
  
  /embeddings:
    post:
      summary: Create embeddings
      operationId: createEmbedding
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EmbeddingRequest'
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EmbeddingResponse'
  
  /key/generate:
    post:
      summary: Generate API key
      operationId: generateKey
      security:
        - MasterKey: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                key_alias:
                  type: string
                team_id:
                  type: string
                models:
                  type: array
                  items:
                    type: string
                max_budget:
                  type: number
      responses:
        '200':
          description: Generated key
          content:
            application/json:
              schema:
                type: object
                properties:
                  key:
                    type: string
                  key_id:
                    type: string

components:
  schemas:
    ChatCompletionRequest:
      type: object
      required:
        - model
        - messages
      properties:
        model:
          type: string
        messages:
          type: array
          items:
            type: object
            properties:
              role:
                type: string
                enum: [system, user, assistant]
              content:
                type: string
        temperature:
          type: number
        max_tokens:
          type: integer
        stream:
          type: boolean
```

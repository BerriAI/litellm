import type { Plugin } from "@opencode-ai/plugin"

type JsonObject = Record<string, unknown>

type LiteLLMModelGroup = {
  model_group: string
  max_input_tokens?: number
  max_output_tokens?: number
  input_cost_per_token?: number
  output_cost_per_token?: number
  mode?: string
  supports_vision?: boolean
  supports_reasoning?: boolean
  supports_function_calling?: boolean
  supported_reasoning_efforts?: string[]
  supported_openai_params?: string[]
}

type OpenCodeModel = {
  name: string
  attachment: boolean
  reasoning: boolean
  temperature: boolean
  tool_call: boolean
  cost?: {
    input: number
    output: number
  }
  limit?: {
    context: number
    input?: number
    output: number
  }
  modalities: {
    input: Array<"text" | "image">
    output: ["text"]
  }
  variants?: Record<string, { reasoningEffort: string }>
}

const PROVIDER_ID = "litellm"

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function optionalBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined
}

function optionalStrings(value: unknown): string[] | undefined {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) return undefined
  return value
}

function parseModelIDs(payload: unknown): string[] {
  if (!isObject(payload) || !Array.isArray(payload.data)) return []

  return Array.from(
    new Set(
      payload.data.flatMap((item) => {
        if (!isObject(item)) return []
        const id = optionalString(item.id)
        return id && !id.includes("*") ? [id] : []
      }),
    ),
  )
}

function parseModelGroups(payload: unknown): Map<string, LiteLLMModelGroup> {
  if (!isObject(payload) || !Array.isArray(payload.data)) return new Map()

  return new Map(
    payload.data.flatMap((item): Array<[string, LiteLLMModelGroup]> => {
      if (!isObject(item)) return []
      const modelGroup = optionalString(item.model_group)
      if (!modelGroup) return []

      return [
        [
          modelGroup,
          {
            model_group: modelGroup,
            max_input_tokens: optionalNumber(item.max_input_tokens),
            max_output_tokens: optionalNumber(item.max_output_tokens),
            input_cost_per_token: optionalNumber(item.input_cost_per_token),
            output_cost_per_token: optionalNumber(item.output_cost_per_token),
            mode: optionalString(item.mode),
            supports_vision: optionalBoolean(item.supports_vision),
            supports_reasoning: optionalBoolean(item.supports_reasoning),
            supports_function_calling: optionalBoolean(item.supports_function_calling),
            supported_reasoning_efforts: optionalStrings(item.supported_reasoning_efforts),
            supported_openai_params: optionalStrings(item.supported_openai_params),
          },
        ],
      ]
    }),
  )
}

function modelConfig(id: string, group?: LiteLLMModelGroup): OpenCodeModel {
  const context = group?.max_input_tokens
  const output = group?.max_output_tokens
  const inputCost = group?.input_cost_per_token
  const outputCost = group?.output_cost_per_token
  const reasoningEfforts = group?.supported_reasoning_efforts

  return {
    name: id,
    attachment: group?.supports_vision ?? false,
    reasoning: group?.supports_reasoning ?? false,
    temperature: group?.supported_openai_params?.includes("temperature") ?? true,
    tool_call: group?.supports_function_calling ?? true,
    ...(inputCost !== undefined && outputCost !== undefined
      ? { cost: { input: inputCost * 1_000_000, output: outputCost * 1_000_000 } }
      : {}),
    ...(context !== undefined && output !== undefined
      ? { limit: { context, output } }
      : {}),
    modalities: {
      input: group?.supports_vision ? ["text", "image"] : ["text"],
      output: ["text"],
    },
    ...(reasoningEfforts?.length
      ? {
          variants: Object.fromEntries(
            reasoningEfforts.map((effort) => [effort, { reasoningEffort: effort }]),
          ),
        }
      : {}),
  }
}

async function getJSON(url: string, apiKey: string): Promise<unknown> {
  const response = await fetch(url, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: "application/json",
    },
    signal: AbortSignal.timeout(10_000),
  })

  if (!response.ok) throw new Error(`${url} returned ${response.status}`)
  return response.json()
}

export const LiteLLM: Plugin = async ({ client }) => {
  const configuredURL = process.env.LITELLM_BASE_URL?.trim()
  const apiKey = process.env.LITELLM_API_KEY?.trim()
  if (!configuredURL || !apiKey) return {}

  const gatewayURL = configuredURL.replace(/\/+$/, "")
  const apiRoot = gatewayURL.endsWith("/v1") ? gatewayURL.slice(0, -3) : gatewayURL
  const inferenceURL = `${apiRoot}/v1`

  try {
    const [modelsPayload, groupsPayload] = await Promise.all([
      getJSON(`${inferenceURL}/models`, apiKey),
      getJSON(`${apiRoot}/model_group/info`, apiKey).catch(() => undefined),
    ])
    const groups = parseModelGroups(groupsPayload)
    const models = Object.fromEntries(
      parseModelIDs(modelsPayload)
        .filter((id) => {
          const mode = groups.get(id)?.mode
          return mode === undefined || mode === "chat"
        })
        .map((id) => [id, modelConfig(id, groups.get(id))]),
    )

    if (Object.keys(models).length === 0) throw new Error("the gateway returned no selectable models")

    return {
      async config(config) {
        config.provider ??= {}
        const existing = config.provider[PROVIDER_ID]
        config.provider[PROVIDER_ID] = {
          npm: "@ai-sdk/openai-compatible",
          name: "LiteLLM Gateway",
          ...existing,
          options: {
            baseURL: inferenceURL,
            apiKey,
            ...existing?.options,
          },
          models: {
            ...models,
            ...existing?.models,
          },
        }
      },
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    await client.app.log({
      body: {
        service: "litellm-model-discovery",
        level: "warn",
        message: `LiteLLM model discovery skipped: ${message}`,
      },
    })
    return {}
  }
}

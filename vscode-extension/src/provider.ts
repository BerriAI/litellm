import * as vscode from "vscode";
import { APIError } from "openai";
import { isResponsesMode, gatewayConfigFrom, summarizeErrorBody, type GatewayClient, type GatewayConfig, type GatewayConfigResult, type ModelGroupsResult } from "./gateway";
import { buildChatCompletionParams, buildResponsesCompletionParams, estimateMessageTokens } from "./messages";
import { describeModels, estimateTokens, reasoningEffortFrom, MODE_CHAT, MODE_RESPONSES, type ModelDescriptor, type ModelGroupInfo } from "./models";
import { responseParts, type ResponsePart } from "./stream";

export interface LiteLLMModel extends vscode.LanguageModelChatInformation {
  readonly gateway: GatewayConfig;
  readonly mode: string;
}

export const TRUNCATED_MESSAGE = "The model stopped at its output token limit before finishing the response";
export const GATEWAY_TIMEOUT_MESSAGE =
  "LiteLLM gateway timed out (HTTP 504). Check the gateway or upstream model and try again.";

const RECONFIGURE_HINT =
  'Fix it from the gear on its row in Manage Language Models: "Update API Key" for the key, "Open in Language Models (JSON)" for the URL';

const configurationProblem = (result: Exclude<GatewayConfigResult, { kind: "ok" | "unconfigured" }>): string => {
  switch (result.kind) {
    case "missing_fields":
      return `LiteLLM provider is missing its ${result.fields.join(" and ")}. ${RECONFIGURE_HINT}`;
    case "invalid_url":
      return `LiteLLM gateway URL "${result.baseUrl}" is not an http or https URL. ${RECONFIGURE_HINT}`;
  }
};

const discoveryFailure = (result: Exclude<ModelGroupsResult, { kind: "ok" }>, baseUrl: string): string => {
  switch (result.kind) {
    case "http_error":
      return `LiteLLM gateway at ${baseUrl} answered ${result.status} for /model_group/info: ${summarizeErrorBody(result.body)}`;
    case "invalid_response":
      return `LiteLLM gateway at ${baseUrl} returned an unexpected /model_group/info payload: ${result.reason}`;
  }
};

const toModel = (descriptor: ModelDescriptor, gateway: GatewayConfig, mode: string): LiteLLMModel => ({
  id: descriptor.id,
  name: descriptor.name,
  family: descriptor.family,
  version: descriptor.version,
  detail: descriptor.detail,
  tooltip: descriptor.tooltip,
  ...(descriptor.inputCost === undefined ? {} : { inputCost: descriptor.inputCost }),
  ...(descriptor.outputCost === undefined ? {} : { outputCost: descriptor.outputCost }),
  ...(descriptor.pricing === undefined ? {} : { pricing: descriptor.pricing }),
  ...(descriptor.priceCategory === undefined ? {} : { priceCategory: descriptor.priceCategory }),
  maxInputTokens: descriptor.maxInputTokens,
  maxOutputTokens: descriptor.maxOutputTokens,
  capabilities: { imageInput: descriptor.imageInput, toolCalling: descriptor.toolCalling },
  ...(descriptor.configurationSchema === undefined ? {} : { configurationSchema: descriptor.configurationSchema }),
  gateway,
  mode,
});

const toVscodePart = (part: ResponsePart): vscode.LanguageModelResponsePart => {
  switch (part.kind) {
    case "text":
      return new vscode.LanguageModelTextPart(part.value);
    case "tool_call":
      return new vscode.LanguageModelToolCallPart(part.callId, part.name, part.input);
    case "invalid_tool_call":
      throw new Error(`Model returned invalid JSON arguments for tool ${part.name}: ${part.arguments}`);
    case "truncated":
      throw new Error(TRUNCATED_MESSAGE);
  }
};

const chatRequestError = (error: unknown): unknown =>
  error instanceof APIError && error.status === 504 ? new Error(GATEWAY_TIMEOUT_MESSAGE) : error;

const withAbortSignal = async <T>(token: vscode.CancellationToken, run: (signal: AbortSignal) => Promise<T>): Promise<T> => {
  const controller = new AbortController();
  const subscription = token.onCancellationRequested(() => controller.abort());
  try {
    return await run(controller.signal);
  } finally {
    subscription.dispose();
  }
};

export class LiteLLMChatProvider implements vscode.LanguageModelChatProvider<LiteLLMModel>, vscode.Disposable {
  private readonly changeEmitter = new vscode.EventEmitter<void>();
  readonly onDidChangeLanguageModelChatInformation = this.changeEmitter.event;

  constructor(private readonly gateway: GatewayClient) {}

  refresh(): void {
    this.changeEmitter.fire();
  }

  dispose(): void {
    this.changeEmitter.dispose();
  }

  async provideLanguageModelChatInformation(
    options: vscode.PrepareLanguageModelChatModelOptions,
    token: vscode.CancellationToken,
  ): Promise<LiteLLMModel[]> {
    const configured = gatewayConfigFrom(options.configuration);
    if (configured.kind === "unconfigured") {
      return [];
    }
    if (configured.kind !== "ok") {
      throw new Error(configurationProblem(configured));
    }
    const result = await withAbortSignal(token, (signal) => this.gateway.listModelGroups(configured.config, signal));
    if (result.kind !== "ok") {
      throw new Error(discoveryFailure(result, configured.config.baseUrl));
    }
    return describeModels(result.groups).map((descriptor) => toModel(descriptor, configured.config, descriptor.mode));
  }

  async provideLanguageModelChatResponse(
    model: LiteLLMModel,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    if (model.mode === MODE_RESPONSES) {
      await this.streamResponses(model, messages, options, progress, token);
    } else {
      await this.streamChat(model, messages, options, progress, token);
    }
  }

  async provideTokenCount(_model: LiteLLMModel, text: string | vscode.LanguageModelChatRequestMessage): Promise<number> {
    return typeof text === "string" ? estimateTokens(text) : estimateMessageTokens(text);
  }

  private async streamChat(
    model: LiteLLMModel,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    const params = buildChatCompletionParams({
      model: model.id,
      messages,
      tools: options.tools ?? [],
      requireToolCall: options.toolMode === vscode.LanguageModelChatToolMode.Required,
      reasoningEffort: reasoningEffortFrom(options.modelConfiguration),
      modelOptions: options.modelOptions ?? {},
    });
    try {
      await withAbortSignal(token, async (signal) => {
        const chunks = await this.gateway.streamChatCompletion(model.gateway, params, signal);
        for await (const part of responseParts(chunks)) {
          progress.report(toVscodePart(part));
        }
      });
    } catch (error) {
      if (token.isCancellationRequested) {
        return;
      }
      throw chatRequestError(error);
    }
  }

  private async streamResponses(
    model: LiteLLMModel,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    const params = buildResponsesCompletionParams({
      model: model.id,
      messages,
      tools: options.tools ?? [],
      requireToolCall: options.toolMode === vscode.LanguageModelChatToolMode.Required,
      reasoningEffort: reasoningEffortFrom(options.modelConfiguration),
      modelOptions: options.modelOptions ?? {},
    });
    try {
      await withAbortSignal(token, async (signal) => {
        const chunks = await this.gateway.streamResponsesCompletion(model.gateway, params, signal);
        for await (const part of responseParts(chunks)) {
          progress.report(toVscodePart(part));
        }
      });
    } catch (error) {
      if (token.isCancellationRequested) {
        return;
      }
      throw chatRequestError(error);
    }
  }
}

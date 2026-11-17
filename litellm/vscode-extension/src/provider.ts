import * as vscode from "vscode";
import { gatewayConfigFrom, summarizeErrorBody, type GatewayClient, type GatewayConfig, type GatewayConfigResult, type ModelGroupsResult } from "./gateway";
import { buildChatCompletionParams, estimateMessageTokens } from "./messages";
import { describeModels, estimateTokens, reasoningEffortFrom, type ModelDescriptor } from "./models";
import { responseParts, type ResponsePart } from "./stream";

export interface LiteLLMModel extends vscode.LanguageModelChatInformation {
  readonly gateway: GatewayConfig;
}

export const TRUNCATED_MESSAGE = "The model stopped at its output token limit before finishing the response";

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

const toModel = (descriptor: ModelDescriptor, gateway: GatewayConfig): LiteLLMModel => ({
  id: descriptor.id,
  name: descriptor.name,
  family: descriptor.family,
  version: descriptor.version,
  detail: descriptor.detail,
  tooltip: descriptor.tooltip,
  maxInputTokens: descriptor.maxInputTokens,
  maxOutputTokens: descriptor.maxOutputTokens,
  capabilities: { imageInput: descriptor.imageInput, toolCalling: descriptor.toolCalling },
  ...(descriptor.configurationSchema === undefined ? {} : { configurationSchema: descriptor.configurationSchema }),
  gateway,
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
    return describeModels(result.groups).map((descriptor) => toModel(descriptor, configured.config));
  }

  async provideLanguageModelChatResponse(
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
      throw error;
    }
  }

  async provideTokenCount(_model: LiteLLMModel, text: string | vscode.LanguageModelChatRequestMessage): Promise<number> {
    return typeof text === "string" ? estimateTokens(text) : estimateMessageTokens(text);
  }
}

import * as vscode from "vscode";
import { createGatewayClient } from "./gateway";
import { LiteLLMChatProvider } from "./provider";

export const VENDOR = "litellm";
export const REFRESH_COMMAND = "litellm.refreshModels";

export function activate(context: vscode.ExtensionContext): void {
  const provider = new LiteLLMChatProvider(createGatewayClient());
  context.subscriptions.push(
    provider,
    vscode.lm.registerLanguageModelChatProvider(VENDOR, provider),
    vscode.commands.registerCommand(REFRESH_COMMAND, () => provider.refresh()),
  );
}

export function deactivate(): void {}

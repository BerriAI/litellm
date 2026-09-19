import type { ConfigurationSchema, ConfigurationValues } from "./models";

declare module "vscode" {
  interface LanguageModelChatInformation {
    readonly configurationSchema?: ConfigurationSchema;
  }

  interface PrepareLanguageModelChatModelOptions {
    readonly configuration?: ConfigurationValues;
  }

  interface ProvideLanguageModelChatResponseOptions {
    readonly modelConfiguration?: ConfigurationValues;
  }
}

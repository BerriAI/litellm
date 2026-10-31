declare module "vscode" {
  interface LanguageModelChatInformation {
    readonly inputCost?: number;
    readonly outputCost?: number;
    readonly pricing?: string;
    readonly priceCategory?: string;
  }
}
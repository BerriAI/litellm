import { describe, expect, it } from "vitest";

import type { CredentialItem } from "@/components/networking";

import { credentialProviderOptions, filterCredentials } from "./credential_filters";

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: "Anthropic",
  azure_ai: "Azure AI Foundry (Studio)",
  openai: "OpenAI",
};

const providerLabel = (provider: string): string => PROVIDER_LABELS[provider] ?? provider;

const credential = (
  credential_name: string,
  custom_llm_provider: string | undefined,
  display_name?: string,
): CredentialItem => ({
  credential_name,
  display_name,
  credential_values: {},
  credential_info: { custom_llm_provider },
});

const names = (credentials: readonly CredentialItem[]): string[] => credentials.map((item) => item.credential_name);

const CREDENTIALS: readonly CredentialItem[] = [
  credential("payments-openai", "openai", "Payments Chat"),
  credential("payments-azure", "azure_ai", "Payments Foundry"),
  credential("support-anthropic", "anthropic"),
  credential("orphan", undefined),
];

const filter = (query: string, providers: readonly string[]): string[] =>
  names(filterCredentials(CREDENTIALS, { query, providers }, providerLabel));

describe("filterCredentials", () => {
  it("returns everything when the query is blank and no provider is chosen", () => {
    expect(filter("", [])).toEqual(["payments-openai", "payments-azure", "support-anthropic", "orphan"]);
  });

  it("matches the display name of a credential whose name does not contain the query", () => {
    expect(filter("chat", [])).toEqual(["payments-openai"]);
  });

  it("matches the credential name even when a display name is set", () => {
    expect(filter("payments-azure", [])).toEqual(["payments-azure"]);
  });

  it("ignores case and surrounding whitespace", () => {
    expect(filter("  PAYMENTS ", [])).toEqual(["payments-openai", "payments-azure"]);
  });

  it("matches both the provider display name and the provider slug", () => {
    expect(filter("foundry", [])).toEqual(["payments-azure"]);
    expect(filter("azure_ai", [])).toEqual(["payments-azure"]);
  });

  it("keeps only the chosen providers, dropping credentials without one", () => {
    expect(filter("", ["azure_ai"])).toEqual(["payments-azure"]);
    expect(filter("", ["openai"])).toEqual(["payments-openai"]);
  });

  it("keeps credentials from any of several chosen providers", () => {
    expect(filter("", ["openai", "anthropic"])).toEqual(["payments-openai", "support-anthropic"]);
  });

  it("applies the query and the providers together", () => {
    expect(filter("payments", ["openai"])).toEqual(["payments-openai"]);
    expect(filter("support", ["openai"])).toEqual([]);
  });
});

describe("credentialProviderOptions", () => {
  it("lists each provider once, labelled, in label order, skipping credentials without one", () => {
    const withDuplicates = [...CREDENTIALS, credential("second-openai", "openai"), credential("blank", "")];

    expect(credentialProviderOptions(withDuplicates, providerLabel)).toEqual([
      { value: "anthropic", label: "Anthropic" },
      { value: "azure_ai", label: "Azure AI Foundry (Studio)" },
      { value: "openai", label: "OpenAI" },
    ]);
  });
});

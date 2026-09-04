import { describe, expect, it, vi } from "vitest";
import { Providers } from "../provider_info_helpers";
import {
  computeCredentialValuesToDelete,
  type CredentialTestInput,
  litellmProviderId,
  planCredentialTest,
  resetCredentialFormOnProviderChange,
  summarizeDiscoveredModels,
} from "./credential_form_helpers";

/**
 * Build a minimal FormInstance stub that records calls. We don't depend
 * on the full Antd API surface — only the three methods the helper uses.
 */
function makeFormStub(initialFields: Record<string, unknown> = {}) {
  const fields: Record<string, unknown> = { ...initialFields };
  const stub = {
    getFieldValue: vi.fn((key: string) => fields[key]),
    setFieldValue: vi.fn((key: string, value: unknown) => {
      fields[key] = value;
    }),
    resetFields: vi.fn(() => {
      Object.keys(fields).forEach((k) => delete fields[k]);
    }),
  };
  return { stub: stub as unknown as Parameters<typeof resetCredentialFormOnProviderChange>[0], fields, calls: stub };
}

describe("resetCredentialFormOnProviderChange", () => {
  it("clears all fields when switching providers", () => {
    // Simulate the OpenAI->Google AI Studio leak: api_base picked up
    // OpenAI's default value and the user typed a custom URL.
    const leakedOpenAiForm = {
      credential_name: "my-prod-key",
      custom_llm_provider: "OpenAI",
      api_base: "https://api.openai.com/v1",
      api_key: "sk-stale-openai-key",
      organization: "org-leak",
    };
    const { stub, fields, calls } = makeFormStub(leakedOpenAiForm);
    const setSelectedProvider = vi.fn();

    resetCredentialFormOnProviderChange(stub, Providers.Google_AI_Studio, setSelectedProvider);

    expect(calls.resetFields).toHaveBeenCalledTimes(1);
    // Provider-specific fields must be gone so the next render starts
    // from the new provider's default_value, not OpenAI's leftover.
    expect(fields.api_base).toBeUndefined();
    expect(fields.api_key).toBeUndefined();
    expect(fields.organization).toBeUndefined();
  });

  it("preserves credential_name across the switch", () => {
    // credential_name is user-supplied metadata, not provider-specific.
    // The admin shouldn't have to retype it just because they re-picked
    // the provider.
    const { stub, fields } = makeFormStub({
      credential_name: "my-prod-key",
      custom_llm_provider: "OpenAI",
      api_base: "https://api.openai.com/v1",
    });

    resetCredentialFormOnProviderChange(stub, Providers.Google_AI_Studio, vi.fn());

    expect(fields.credential_name).toBe("my-prod-key");
  });

  it("updates custom_llm_provider and selectedProvider state to the new value", () => {
    const { stub, fields } = makeFormStub({ credential_name: "x" });
    const setSelectedProvider = vi.fn();

    resetCredentialFormOnProviderChange(stub, Providers.Google_AI_Studio, setSelectedProvider);

    expect(fields.custom_llm_provider).toBe(Providers.Google_AI_Studio);
    expect(setSelectedProvider).toHaveBeenCalledExactlyOnceWith(Providers.Google_AI_Studio);
  });

  it("does not call setFieldValue('credential_name', undefined) when the name was unset", () => {
    // Edge case: brand-new modal with no name typed yet. We shouldn't
    // explicitly write `undefined` back into the form (Antd treats that
    // as a touched empty field, triggering the "required" validation
    // prematurely).
    const { stub, calls } = makeFormStub({});

    resetCredentialFormOnProviderChange(stub, Providers.Anthropic, vi.fn());

    const credentialNameCalls = calls.setFieldValue.mock.calls.filter(([key]) => key === "credential_name");
    expect(credentialNameCalls).toHaveLength(0);
  });
});

describe("computeCredentialValuesToDelete", () => {
  it("flags a field that is no longer mounted at all", () => {
    // e.g. switching from the api_key variant to a WIF variant unmounts api_key/api_base.
    const original = { api_base: "https://api.anthropic.com", api_key: "sk-***1234" };
    const mounted = { anthropic_federation_rule_id: "rule-1" };

    expect(computeCredentialValuesToDelete(original, mounted)).toEqual(["api_base", "api_key"]);
  });

  it("keeps a masked-but-untouched field, since it is still mounted", () => {
    const original = { api_key: "sk-***1234" };
    const mounted = { api_key: "sk-***1234" };

    expect(computeCredentialValuesToDelete(original, mounted)).toEqual([]);
  });

  it("deletes a field the operator cleared, since the caller drops it from the payload", () => {
    const original = { api_base: "https://old.gateway.internal" };
    const mounted = { api_base: "" };

    expect(computeCredentialValuesToDelete(original, mounted)).toEqual(["api_base"]);
  });

  it("keeps a field the operator genuinely changed", () => {
    const original = { api_key: "sk-***1234" };
    const mounted = { api_key: "sk-new-real-key" };

    expect(computeCredentialValuesToDelete(original, mounted)).toEqual([]);
  });

  it("returns nothing when nothing existed before", () => {
    expect(computeCredentialValuesToDelete({}, { api_key: "sk-new" })).toEqual([]);
  });
});

describe("litellmProviderId", () => {
  it("maps a dashboard provider key to the litellm provider id", () => {
    expect(litellmProviderId("Anthropic")).toBe("anthropic");
    expect(litellmProviderId("Google_AI_Studio")).toBe("gemini");
  });

  it("passes a litellm provider id stored by curl or the API through unchanged", () => {
    expect(litellmProviderId("anthropic")).toBe("anthropic");
  });
});

describe("planCredentialTest", () => {
  const planWith = (overrides: Partial<CredentialTestInput>) => {
    const input: CredentialTestInput = {
      mode: "add",
      provider: "OpenAI",
      credentialName: "",
      mountedValues: {},
      hasUnsavedChanges: true,
      ...overrides,
    };
    return planCredentialTest(input);
  };
  const addWith = (mountedValues: Record<string, unknown>) => planWith({ mountedValues });

  it("sends only the entered api_key and api_base inline before saving", () => {
    const keyOnlyForm = { credential_name: "prod", custom_llm_provider: "OpenAI", api_key: "sk-x", api_base: "" };
    expect(addWith(keyOnlyForm)).toEqual({
      kind: "ready",
      request: { custom_llm_provider: "openai", api_key: "sk-x" },
    });
    expect(addWith({ api_key: "sk-x", api_base: "https://proxy.example.com/v1" })).toEqual({
      kind: "ready",
      request: { custom_llm_provider: "openai", api_key: "sk-x", api_base: "https://proxy.example.com/v1" },
    });
  });

  it("asks for values before anything has been entered", () => {
    expect(addWith({ credential_name: "prod", custom_llm_provider: "OpenAI", api_key: null })).toEqual({
      kind: "unavailable",
      reason: "Fill in the credential values first.",
    });
  });

  it("refuses to send server-owned values inline and points at saving first", () => {
    const plan = planWith({
      provider: "Anthropic",
      mountedValues: { anthropic_identity_source: "internal_issuer", anthropic_issuer_url: "", api_base: "" },
    });
    expect(plan.kind).toBe("unavailable");
    expect(plan.kind === "unavailable" && plan.reason).toMatch(/^Add the credential first/);
  });

  it("tests a saved credential by name so its values never leave the proxy", () => {
    const savedWif: Partial<CredentialTestInput> = {
      mode: "edit",
      provider: "Anthropic",
      credentialName: "anthropic-wif",
      mountedValues: { anthropic_federation_rule_id: "fdrl_1", anthropic_organization_id: "org-1" },
      hasUnsavedChanges: false,
    };
    expect(planWith(savedWif)).toEqual({
      kind: "ready",
      request: { custom_llm_provider: "anthropic", litellm_credential_name: "anthropic-wif" },
    });
  });

  it("requires unsaved edits to be saved before a by-name test", () => {
    const editedWif: Partial<CredentialTestInput> = {
      mode: "edit",
      provider: "Anthropic",
      credentialName: "anthropic-wif",
      mountedValues: { api_key: "sk-new" },
    };
    expect(planWith(editedWif)).toEqual({
      kind: "unavailable",
      reason: "Update the credential first. Test Connection checks the saved values.",
    });
  });
});

describe("summarizeDiscoveredModels", () => {
  it("reports an empty discovery as a success without models", () => {
    expect(summarizeDiscoveredModels([])).toBe("Connection succeeded, but the provider returned no models.");
  });

  it("lists up to three models and counts the rest", () => {
    expect(summarizeDiscoveredModels(["a"])).toBe("Connection succeeded. 1 model available: a.");
    expect(summarizeDiscoveredModels(["a", "b", "c", "d", "e"])).toBe(
      "Connection succeeded. 5 models available: a, b, c and 2 more.",
    );
  });
});

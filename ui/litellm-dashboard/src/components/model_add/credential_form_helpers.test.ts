import type { FormInstance } from "antd";
import { describe, expect, it, vi } from "vitest";
import { Providers } from "../provider_info_helpers";
import { resetCredentialFormOnProviderChange } from "./credential_form_helpers";

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
  return { stub: stub as unknown as FormInstance, fields, calls: stub };
}

describe("resetCredentialFormOnProviderChange", () => {
  it("clears all fields when switching providers", () => {
    // Simulate the OpenAI->Google AI Studio leak: api_base picked up
    // OpenAI's default value and the user typed a custom URL.
    const { stub, fields, calls } = makeFormStub({
      credential_name: "my-prod-key",
      custom_llm_provider: "OpenAI",
      api_base: "https://api.openai.com/v1",
      api_key: "sk-stale-openai-key",
      organization: "org-leak",
    });
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

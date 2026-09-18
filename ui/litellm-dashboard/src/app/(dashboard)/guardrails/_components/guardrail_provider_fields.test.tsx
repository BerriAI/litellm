import React from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { useForm } from "react-hook-form";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailProviderFields from "./guardrail_provider_fields";
import { populateGuardrailProviderMap } from "./guardrail_info_helpers";
import type { GuardrailFormValues } from "./GuardrailFormField";

vi.mock("@/lib/toast", () => ({ toast: { error: vi.fn() } }));

const HIDE_SECRETS_PARAMS = {
  "hide-secrets": {
    ui_friendly_name: "Hide Secrets",
    detect_secrets_config: {
      param: "detect_secrets_config",
      description: "Optional detect-secrets configuration",
      required: false,
      type: "object",
    },
  },
};

const Harness: React.FC<{ onValid: (values: GuardrailFormValues) => void }> = ({ onValid }) => {
  const form = useForm<GuardrailFormValues>();
  return (
    <form onSubmit={form.handleSubmit(onValid)}>
      <GuardrailProviderFields
        selectedProvider="Hide-secrets"
        control={form.control}
        providerParams={HIDE_SECRETS_PARAMS}
      />
      <button type="submit">save</button>
    </form>
  );
};

const renderHarness = () => {
  populateGuardrailProviderMap(HIDE_SECRETS_PARAMS);
  const onValid = vi.fn();
  renderWithProviders(<Harness onValid={onValid} />);
  const textarea = screen.getByLabelText(/detect_secrets_config/) as HTMLTextAreaElement;
  return { onValid, textarea };
};

describe("GuardrailProviderFields object field", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("commits a valid JSON object as a parsed dict", async () => {
    const { onValid, textarea } = renderHarness();

    fireEvent.change(textarea, { target: { value: '{"plugins_used": [{"name": "AWSKeyDetector"}]}' } });
    fireEvent.blur(textarea);
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    await waitFor(() => expect(onValid).toHaveBeenCalledTimes(1));
    expect(onValid.mock.calls[0][0].detect_secrets_config).toEqual({
      plugins_used: [{ name: "AWSKeyDetector" }],
    });
  });

  it("blocks submission while the field holds malformed JSON", async () => {
    const { onValid, textarea } = renderHarness();

    fireEvent.change(textarea, { target: { value: "{not json" } });
    fireEvent.blur(textarea);
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    await screen.findByText("detect_secrets_config must be a valid JSON object");
    expect(onValid).not.toHaveBeenCalled();
    expect(textarea.value).toBe("{not json");
  });

  it.each(['["array"]', '"scalar"', "null", "42"])("blocks non-object JSON %s", async (raw) => {
    const { onValid, textarea } = renderHarness();

    fireEvent.change(textarea, { target: { value: raw } });
    fireEvent.blur(textarea);
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    await screen.findByText("detect_secrets_config must be a valid JSON object");
    expect(onValid).not.toHaveBeenCalled();
  });

  it("treats a cleared field as unset and submits", async () => {
    const { onValid, textarea } = renderHarness();

    fireEvent.change(textarea, { target: { value: '{"a": 1}' } });
    fireEvent.blur(textarea);
    fireEvent.change(textarea, { target: { value: "" } });
    fireEvent.blur(textarea);
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    await waitFor(() => expect(onValid).toHaveBeenCalledTimes(1));
    expect(onValid.mock.calls[0][0].detect_secrets_config).toBeUndefined();
  });
});

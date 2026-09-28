import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { chooseSelectOption, renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddGuardrailForm from "./add_guardrail_form";

vi.mock("@/components/networking", () => ({
  createGuardrailCall: vi.fn(),
  getGuardrailProviderSpecificParams: vi.fn(),
  getGuardrailUISettings: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

import * as networking from "@/components/networking";

const providerParams = {
  bedrock: {
    ui_friendly_name: "Bedrock Guardrail",
    guardrailIdentifier: { description: "The guardrail id on Bedrock", required: true, type: null },
    api_key: { description: "Bedrock API key", required: false, type: null },
    optional_params: {
      description: "Optional parameters",
      required: false,
      type: "nested",
      fields: {
        severity_threshold: { description: "Severity threshold", required: false, type: "number" },
      },
    },
  },
  llm_as_a_judge: {
    ui_friendly_name: "LiteLLM LLM as a Judge",
  },
};

const uiSettings = {
  supported_entities: [],
  supported_actions: [],
  supported_modes: ["pre_call", "post_call"],
  pii_entity_categories: [],
};

const renderForm = () => {
  const onSuccess = vi.fn();
  const onClose = vi.fn();
  renderWithProviders(
    <AddGuardrailForm visible onClose={onClose} accessToken="test-token" onSuccess={onSuccess} preset={undefined} />,
  );
  return { onSuccess, onClose };
};

const pickProvider = async (user: ReturnType<typeof userEvent.setup>, label: string) => {
  await user.click(screen.getByLabelText("Guardrail Provider"));
  await user.click(await screen.findByText(label));
};

const payload = () => vi.mocked(networking.createGuardrailCall).mock.calls.at(-1)?.[1];

describe("AddGuardrailForm create payload characterization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getGuardrailUISettings).mockResolvedValue(uiSettings);
    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue(providerParams);
    vi.mocked(networking.modelAvailableCall).mockResolvedValue({ data: [{ id: "gpt-5" }] });
    vi.mocked(networking.createGuardrailCall).mockResolvedValue({ guardrail_id: "new" });
  });

  it("sends the seeded step-0 defaults even though those fields are unmounted at submit time", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "my-bedrock");
    await pickProvider(user, "Bedrock Guardrail");
    await user.type(await screen.findByPlaceholderText("The guardrail id on Bedrock"), "gr-123");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(networking.createGuardrailCall).toHaveBeenCalledTimes(1));
    expect(payload()).toEqual({
      guardrail_name: "my-bedrock",
      litellm_params: {
        guardrail: "bedrock",
        mode: "pre_call",
        default_on: false,
        guardrailIdentifier: "gr-123",
      },
      guardrail_info: {},
    });
  });

  it("switches mode from the seeded string to an array once the user touches the multi select", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "my-bedrock");
    await pickProvider(user, "Bedrock Guardrail");
    await user.click(screen.getByLabelText("Mode"));
    const postCallOption = (await screen.findAllByText("post_call")).at(-1) as HTMLElement;
    await user.click(postCallOption);
    await user.type(await screen.findByPlaceholderText("The guardrail id on Bedrock"), "gr-123");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(networking.createGuardrailCall).toHaveBeenCalledTimes(1));
    expect(payload()).toMatchObject({ litellm_params: { mode: ["pre_call", "post_call"] } });
  });

  it("blocks Next when the user deselects every mode", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "my-bedrock");
    await pickProvider(user, "Bedrock Guardrail");
    await user.click(screen.getByLabelText("Mode"));
    const preCallOption = (await screen.findAllByText("pre_call")).at(-1) as HTMLElement;
    await user.click(preCallOption);
    await user.type(await screen.findByPlaceholderText("The guardrail id on Bedrock"), "gr-123");
    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByText("Please select a mode")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Guardrail" })).not.toBeInTheDocument();
  });

  it("copies a value typed in the optional params step up to the top level of litellm_params", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "my-bedrock");
    await pickProvider(user, "Bedrock Guardrail");
    await user.type(await screen.findByPlaceholderText("The guardrail id on Bedrock"), "gr-123");
    await user.click(screen.getByRole("button", { name: "Next" }));

    await user.type(await screen.findByPlaceholderText("Severity threshold"), "4");
    await user.click(screen.getByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(networking.createGuardrailCall).toHaveBeenCalledTimes(1));
    expect(payload()).toMatchObject({
      litellm_params: { guardrailIdentifier: "gr-123", severity_threshold: 4 },
    });
  });

  it("omits a provider param the user left blank rather than sending an empty string", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "my-bedrock");
    await pickProvider(user, "Bedrock Guardrail");
    await user.type(await screen.findByPlaceholderText("The guardrail id on Bedrock"), "gr-123");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(networking.createGuardrailCall).toHaveBeenCalledTimes(1));
    expect(payload()).not.toHaveProperty("litellm_params.api_key");
  });

  it("blocks Next and sends nothing when the required guardrail name is missing", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await pickProvider(user, "Bedrock Guardrail");
    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByText("Please enter a guardrail name")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Guardrail" })).not.toBeInTheDocument();
  });

  it("creates the guardrail even though a required provider field was left blank", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "my-bedrock");
    await pickProvider(user, "Bedrock Guardrail");
    await screen.findByPlaceholderText("The guardrail id on Bedrock");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(networking.createGuardrailCall).toHaveBeenCalledTimes(1));
    expect(payload()).toEqual({
      guardrail_name: "my-bedrock",
      litellm_params: { guardrail: "bedrock", mode: "pre_call", default_on: false },
      guardrail_info: {},
    });
  });

  it("keeps a step-0 value the user typed after moving forward and back", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "round-trip");
    await pickProvider(user, "Bedrock Guardrail");
    await user.type(await screen.findByPlaceholderText("The guardrail id on Bedrock"), "gr-123");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Previous" }));

    expect(await screen.findByLabelText("Guardrail Name")).toHaveValue("round-trip");
  });

  it("sends the llm judge criteria with numeric weights and the seeded threshold default", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "judge-1");
    await pickProvider(user, "LiteLLM LLM as a Judge");
    await user.click(screen.getByRole("button", { name: "Next" }));

    await user.type(await screen.findByPlaceholderText("Criterion name (e.g. Policy accuracy)"), "Accuracy");
    await user.type(screen.getByPlaceholderText("What should the judge check for this criterion?"), "Is it right");
    await user.click(screen.getByLabelText("Judge Model"));
    await user.click(await screen.findByTitle("gpt-5"));
    await user.click(screen.getByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(networking.createGuardrailCall).toHaveBeenCalledTimes(1));
    expect(payload()).toEqual({
      guardrail_name: "judge-1",
      litellm_params: {
        guardrail: "llm_as_a_judge",
        mode: "post_call",
        default_on: false,
        judge_model: "gpt-5",
        overall_threshold: 80,
        on_failure: "block",
        criteria: [{ name: "Accuracy", weight: 100, description: "Is it right" }],
      },
      guardrail_info: {},
    });
  });

  it("refuses to create an llm judge guardrail whose criterion weights do not total 100", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "judge-2");
    await pickProvider(user, "LiteLLM LLM as a Judge");
    await user.click(screen.getByRole("button", { name: "Next" }));

    await user.type(await screen.findByPlaceholderText("Criterion name (e.g. Policy accuracy)"), "Accuracy");
    await user.type(screen.getByPlaceholderText("What should the judge check for this criterion?"), "Is it right");
    const weight = screen.getByPlaceholderText("e.g. 50");
    await user.clear(weight);
    await user.type(weight, "60");
    await user.click(screen.getByLabelText("Judge Model"));
    await user.click(await screen.findByTitle("gpt-5"));
    await user.click(screen.getByRole("button", { name: "Create Guardrail" }));

    await screen.findByText(/Weights total: 60%/);
    expect(networking.createGuardrailCall).not.toHaveBeenCalled();
  });
  it("carries the three step-0 selects to the payload when the user moves each off its default", async () => {
    const user = userEvent.setup({ delay: null });
    renderForm();

    await user.type(await screen.findByLabelText("Guardrail Name"), "my-bedrock");
    await pickProvider(user, "Bedrock Guardrail");

    await chooseSelectOption(user, screen.getByLabelText("Always On"), "Yes");
    await chooseSelectOption(
      user,
      screen.getByLabelText("Skip system messages in guardrail"),
      /exclude from guardrail scan/,
    );
    await chooseSelectOption(user, screen.getByLabelText("Skip tool messages in guardrail"), /always include in scan/);

    await user.type(await screen.findByPlaceholderText("The guardrail id on Bedrock"), "gr-123");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(networking.createGuardrailCall).toHaveBeenCalledTimes(1));
    expect(payload()).toMatchObject({
      litellm_params: {
        default_on: true,
        skip_system_message_in_guardrail: true,
        skip_tool_message_in_guardrail: false,
      },
    });
  });
});

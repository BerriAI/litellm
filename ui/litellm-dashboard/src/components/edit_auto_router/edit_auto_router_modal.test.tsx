import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

import NotificationsManager from "@/components/molecules/notifications_manager";
import EditAutoRouterModal from "./edit_auto_router_modal";

const { modelPatchUpdateCall, modelAvailableCall, getAutoRouterClassifierDefaultPromptCall } = vi.hoisted(() => ({
  modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  getAutoRouterClassifierDefaultPromptCall: vi.fn().mockResolvedValue("Classify the request into exactly one tier."),
}));

vi.mock("../networking", () => ({
  modelPatchUpdateCall,
  modelAvailableCall,
  getAutoRouterClassifierDefaultPromptCall,
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "sk-test" }) }));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-4o-mini" }]),
}));

const STORED_CONFIG = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o-mini"], COMPLEX: ["gpt-4o-mini"], REASONING: ["gpt-4o-mini"] },
  classifier_type: "heuristic",
  keyword_tier_rules: [{ keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  escalation_keywords: ["urgent", "outage"],
  semantic_keyword_matching: true,
  embedding_model: "voyage-4-large",
  match_threshold: 0.72,
};

const MODEL_DATA = {
  model_name: "tri-tier-router",
  litellm_params: {
    model: "auto_router/complexity_router",
    complexity_router_config: STORED_CONFIG,
  },
  model_info: { id: "auto-1", access_groups: [] },
};

const renderModal = () =>
  renderWithProviders(
    <EditAutoRouterModal
      isVisible
      onCancel={vi.fn()}
      onSuccess={vi.fn()}
      modelData={MODEL_DATA}
      accessToken="token"
      userRole="Admin"
    />,
  );

const savedConfig = () => {
  const [, payload] = modelPatchUpdateCall.mock.calls.at(-1) ?? [];
  return payload?.litellm_params?.complexity_router_config;
};

describe("EditAutoRouterModal keyword matching", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  it("renders the advanced sections the create form offers", async () => {
    renderModal();

    expect(await screen.findByText(/Escalation Keywords/i)).toBeInTheDocument();
    expect(await screen.findByText(/Keyword\/Semantic Matching/i)).toBeInTheDocument();
  });

  // These keys are rewritten from form state on save, so if the modal renders the controls
  // without hydrating them, an untouched save silently wipes the stored configuration. This
  // drives the real component; a test of the payload builder alone cannot see that bug.
  it("preserves stored keyword matching through an untouched open-and-save", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText(/Escalation Keywords/i);
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());

    const config = savedConfig();
    expect(config.keyword_tier_rules).toEqual([{ keywords: ["invoice", "refund"], tier: "MEDIUM" }]);
    expect(config.escalation_keywords).toEqual(["urgent", "outage"]);
    expect(config.semantic_keyword_matching).toBe(true);
    expect(config.embedding_model).toBe("voyage-4-large");
    expect(config.match_threshold).toBe(0.72);
  });

  // The create form blocks this; the edit modal renders the same controls, so it must block it
  // too. The backend raises on semantic_keyword_matching without an embedding model or keyword
  // rules, so skipping the guard turns a friendly inline message into a raw 400.
  it("blocks a save that enables semantic matching with no embedding model", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: {
            ...MODEL_DATA.litellm_params,
            complexity_router_config: {
              ...STORED_CONFIG,
              semantic_keyword_matching: true,
              embedding_model: undefined,
              keyword_tier_rules: [{ keywords: ["invoice"], tier: "MEDIUM" }],
            },
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

    await screen.findByText(/Escalation Keywords/i);
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(NotificationsManager.fromBackend).toHaveBeenCalled());
    expect(modelPatchUpdateCall).not.toHaveBeenCalled();
  });

  // LIT-5133, edit side. Semantic matching is off here on purpose: it used to be the only thing
  // that checked a rule for keywords, so with it on this save was already blocked and the test
  // would pass without the fix. Off, the unfilled row was dropped and the save reported success.
  it("blocks a save that adds a keyword rule and leaves it empty", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: {
            ...MODEL_DATA.litellm_params,
            complexity_router_config: {
              ...STORED_CONFIG,
              semantic_keyword_matching: false,
              embedding_model: undefined,
            },
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

    await screen.findByText(/Escalation Keywords/i);
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));

    // The modal renders the same controls as the create form, so it owes the same treatment:
    // the row says what is missing and the save is not offered while it is.
    expect(await screen.findByText("At least one keyword is required")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /save changes/i })).toBeDisabled();
    expect(modelPatchUpdateCall).not.toHaveBeenCalled();
  });

  it("gives the save back once the added keyword rule is filled", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: {
            ...MODEL_DATA.litellm_params,
            complexity_router_config: {
              ...STORED_CONFIG,
              semantic_keyword_matching: false,
              embedding_model: undefined,
            },
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

    await screen.findByText(/Escalation Keywords/i);
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    expect(screen.getByRole("button", { name: /save changes/i })).toBeDisabled();

    await user.type(
      within(screen.getByText("Keywords 2").closest("div") as HTMLElement).getByRole("combobox"),
      "chargeback{enter}",
    );

    expect(screen.getByRole("button", { name: /save changes/i })).toBeEnabled();
    expect(screen.queryByText("At least one keyword is required")).not.toBeInTheDocument();
  });
});

describe("EditAutoRouterModal classifier context window", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  const STORED_LLM_CONFIG = {
    tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o-mini"], COMPLEX: ["gpt-4o-mini"], REASONING: ["gpt-4o-mini"] },
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
    classifier_context_window_size: 5,
    classifier_context_per_turn_chars: 300,
  };

  const renderLlmModal = () =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: { ...MODEL_DATA.litellm_params, complexity_router_config: STORED_LLM_CONFIG },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  // Hydration bugs are invisible to the payload-builder unit tests, which only exercise
  // buildUpdatedComplexityRouterConfig with a form value the caller already assembled by hand.
  // Only driving the real component through open, then save with nothing touched, catches a
  // missing initializeForm hydration line.
  it("shows the stored classifier context values and preserves them through an untouched open-and-save", async () => {
    const user = userEvent.setup();
    renderLlmModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    await screen.findByText("Context Window Size");
    expect(screen.getByDisplayValue("5")).toBeInTheDocument();
    expect(screen.getByDisplayValue("300")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    const config = savedConfig();
    expect(config.classifier_context_window_size).toBe(5);
    expect(config.classifier_context_per_turn_chars).toBe(300);
  });

  // The prompt editor is a base-ui Dialog at z-index 50. Housing this form in an antd Modal put a
  // z-index 1000 overlay between the operator and it, so the editor opened underneath and could
  // not be read or typed into. jsdom does not paint, so the assertion is the invariant behind the
  // stacking: both overlays come from the one Dialog primitive the create form already uses.
  it("opens the classifier prompt editor in the same overlay layer as the form", async () => {
    const user = userEvent.setup();
    const { baseElement } = renderLlmModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    await user.click(await screen.findByRole("button", { name: /prompt/i }));

    expect(await screen.findByLabelText("Classifier system prompt")).toBeInTheDocument();
    expect(baseElement.querySelectorAll('[data-slot="dialog-content"]')).toHaveLength(2);
    expect(baseElement.querySelector(".ant-modal")).toBeNull();
  });

  it("persists an edited classifier context window size", async () => {
    const user = userEvent.setup();
    renderLlmModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    const windowSizeSection = (await screen.findByText("Context Window Size")).closest("div") as HTMLElement;
    const input = within(windowSizeSection).getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "8" } });

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().classifier_context_window_size).toBe(8);
  });
});

describe("EditAutoRouterModal assistant turns", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  const STORED_CONFIG = {
    tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o-mini"], COMPLEX: ["gpt-4o-mini"], REASONING: ["gpt-4o-mini"] },
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
    classifier_context_include_assistant_turns: true,
  };

  const renderModal = () =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: { ...MODEL_DATA.litellm_params, complexity_router_config: STORED_CONFIG },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  // The create and edit stacks share the rendered control but duplicate the serializer, the
  // hydrator and the managed-key set, so a field wired into only one of them fails here and
  // nowhere else: the payload-builder unit tests are handed a form value assembled by hand.
  it("shows the stored value and preserves it through an untouched open-and-save", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    await screen.findByText("Include Assistant Turns");
    expect(screen.getByRole("switch", { name: "Include Assistant Turns" })).toBeChecked();

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().classifier_context_include_assistant_turns).toBe(true);
  });

  it("persists turning assistant turns off", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    await screen.findByText("Include Assistant Turns");
    await user.click(screen.getByRole("switch", { name: "Include Assistant Turns" }));

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().classifier_context_include_assistant_turns).toBe(false);
  });
});

describe("EditAutoRouterModal session affinity", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  const renderWithStoredConfig = (complexity_router_config: Record<string, unknown>) =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{ ...MODEL_DATA, litellm_params: { ...MODEL_DATA.litellm_params, complexity_router_config } }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  // A stored config with no session_affinity key now runs with affinity OFF, because the backend
  // field defaults to False. The toggle has to render what the router actually does, and an
  // untouched save must not flip it.
  it("shows a stored config with no session_affinity key as off", async () => {
    const user = userEvent.setup();
    renderWithStoredConfig(STORED_CONFIG);

    await user.click(await screen.findByText("Advanced: Affinity"));
    expect(await screen.findByRole("switch", { name: "Pin a session to its first model" })).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().session_affinity).toBe(false);
  });

  it("shows a stored session_affinity=true as on and preserves it through an untouched save", async () => {
    const user = userEvent.setup();
    renderWithStoredConfig({ ...STORED_CONFIG, session_affinity: true });

    await user.click(await screen.findByText("Advanced: Affinity"));
    expect(await screen.findByRole("switch", { name: "Pin a session to its first model" })).toBeChecked();

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().session_affinity).toBe(true);
  });

  it("persists turning session affinity on", async () => {
    const user = userEvent.setup();
    renderWithStoredConfig(STORED_CONFIG);

    await user.click(await screen.findByText("Advanced: Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to its first model" }));

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().session_affinity).toBe(true);
  });

  it("persists turning session affinity back off", async () => {
    const user = userEvent.setup();
    renderWithStoredConfig({ ...STORED_CONFIG, session_affinity: true });

    await user.click(await screen.findByText("Advanced: Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to its first model" }));

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().session_affinity).toBe(false);
  });
});

describe("EditAutoRouterModal deployment affinity", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  const renderWithStoredConfig = (complexity_router_config: Record<string, unknown>) =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{ ...MODEL_DATA, litellm_params: { ...MODEL_DATA.litellm_params, complexity_router_config } }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  it("shows a stored config with no deployment_affinity key as on, matching the backend default", async () => {
    const user = userEvent.setup();
    renderWithStoredConfig(STORED_CONFIG);

    await user.click(await screen.findByText("Advanced: Affinity"));
    expect(
      await screen.findByRole("switch", { name: "Pin a session to one deployment per model group" }),
    ).toBeChecked();

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().deployment_affinity).toBe(true);
  });

  it("shows a stored deployment_affinity=false as off and preserves it through an untouched save", async () => {
    const user = userEvent.setup();
    renderWithStoredConfig({ ...STORED_CONFIG, deployment_affinity: false });

    await user.click(await screen.findByText("Advanced: Affinity"));
    expect(
      await screen.findByRole("switch", { name: "Pin a session to one deployment per model group" }),
    ).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().deployment_affinity).toBe(false);
  });

  it("persists turning deployment affinity off", async () => {
    const user = userEvent.setup();
    renderWithStoredConfig(STORED_CONFIG);

    await user.click(await screen.findByText("Advanced: Affinity"));
    await user.click(await screen.findByRole("switch", { name: "Pin a session to one deployment per model group" }));

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().deployment_affinity).toBe(false);
  });
});

describe("EditAutoRouterModal custom classifier prompt and fallback", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  const STORED_CUSTOM_CONFIG = {
    tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o-mini"], COMPLEX: ["gpt-4o-mini"], REASONING: ["gpt-4o-mini"] },
    classifier_type: "llm",
    classifier_llm_config: {
      model: "gpt-4o-mini",
      timeout_ms: 3000,
      system_prompt: "Grade data sensitivity, not difficulty.",
    },
    classifier_fallback: "default_model",
  };

  const renderCustomModal = () =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: { ...MODEL_DATA.litellm_params, complexity_router_config: STORED_CUSTOM_CONFIG },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  // Both keys are rewritten from form state on save, so a missing hydration line would silently
  // wipe an operator's custom prompt the first time they opened this modal for anything else.
  it("preserves a stored custom prompt and fallback through an untouched open-and-save", async () => {
    const user = userEvent.setup();
    renderCustomModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    expect(await screen.findByRole("button", { name: "Edit custom prompt" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).toHaveAttribute("checked");

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    const config = savedConfig();
    expect(config.classifier_llm_config.system_prompt).toBe("Grade data sensitivity, not difficulty.");
    expect(config.classifier_fallback).toBe("default_model");
  });

  it("persists a switch back to the heuristic fallback", async () => {
    const user = userEvent.setup();
    renderCustomModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    await user.click(await screen.findByRole("radio", { name: /Score with the heuristic/ }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().classifier_fallback).toBe("heuristic");
  });

  it("drops the override when the prompt is reset to the default", async () => {
    const user = userEvent.setup();
    renderCustomModal();

    await user.click(await screen.findByText("Advanced: Classification Method"));
    await user.click(await screen.findByRole("button", { name: "Reset to default" }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig().classifier_llm_config).not.toHaveProperty("system_prompt");
  });
});

import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

import { toast } from "@/lib/toast";
import EditAutoRouterModal from "./edit_auto_router_modal";
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

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

    await waitFor(() => expect(toast.fromError).toHaveBeenCalled());
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
      "chargeback",
    );
    await user.click(await screen.findByText('Create "chargeback"'));

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
    expect(screen.queryByText("Context Per-Turn Character Limit")).not.toBeInTheDocument();

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
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).toBeChecked();

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

describe("EditAutoRouterModal default model", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  const savedDefaultModel = () => {
    const [, payload] = modelPatchUpdateCall.mock.calls.at(-1) ?? [];
    return payload?.litellm_params?.complexity_router_default_model;
  };

  const renderWithStoredPin = (default_model?: string) =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: {
            ...MODEL_DATA.litellm_params,
            complexity_router_config: { ...STORED_CONFIG, ...(default_model && { default_model }) },
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  // No config blob marker — only litellm_params.complexity_router_default_model, as an untouched
  // router looked before this PR's marker existed, or one an external API call wrote directly to.
  const renderWithLitellmParamsDefaultOnly = (complexityRouterDefaultModel: string) =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: {
            ...MODEL_DATA.litellm_params,
            complexity_router_config: STORED_CONFIG,
            complexity_router_default_model: complexityRouterDefaultModel,
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  it("preserves a stored pin through an untouched open-and-save", async () => {
    const user = userEvent.setup();
    renderWithStoredPin("out-of-band-default");

    await user.click(await screen.findByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedDefaultModel()).toBe("out-of-band-default");
    expect(savedConfig()).toMatchObject({ default_model: "out-of-band-default" });
  });

  it("shows a stored pin as the selection, so the saved value is not a hidden one", async () => {
    renderWithStoredPin("out-of-band-default");

    const select = await screen.findByRole("combobox", { name: "Default model" });
    expect(select).toHaveValue("out-of-band-default");
  });

  // The pin is recorded in the config rather than inferred by comparing the stored default to a
  // re-derivation, so pinning the model the tiers already imply still reads back as a pin.
  it("keeps a pin that matches what the tiers derive", async () => {
    const user = userEvent.setup();
    renderWithStoredPin(STORED_CONFIG.tiers.MEDIUM[0]);

    const select = await screen.findByRole("combobox", { name: "Default model" });
    expect(select).toHaveValue(STORED_CONFIG.tiers.MEDIUM[0]);

    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig()).toMatchObject({ default_model: STORED_CONFIG.tiers.MEDIUM[0] });
  });

  // Greptile P1 on #36615: with no config blob marker, a litellm_params default that merely
  // matches what the tiers derive is indistinguishable from the pre-PR auto-derive-and-write
  // behavior (main always wrote a tier-derived value there on every save). Treating it as a pin
  // would freeze every pre-existing router's default away from its tiers, so it stays unpinned.
  it("treats a litellm_params default matching tier-derivation as unpinned, not a frozen-in pin", async () => {
    const user = userEvent.setup();
    renderWithLitellmParamsDefaultOnly(STORED_CONFIG.tiers.MEDIUM[0]);

    const select = await screen.findByRole("combobox", { name: "Default model" });
    expect(select).toHaveValue("");

    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig()).not.toHaveProperty("default_model");
    expect(savedDefaultModel()).toBe(STORED_CONFIG.tiers.MEDIUM[0]);
  });

  // Greptile P1 on #36615: a litellm_params default that diverges from tier-derivation could only
  // have gotten there via an explicit override — set by the API directly, since this UI's own
  // save path keeps it in sync with tiers whenever there's no pin. That divergence must survive
  // the next save instead of being silently recomputed away.
  it("treats a diverging litellm_params default as an external pin and preserves it", async () => {
    const user = userEvent.setup();
    renderWithLitellmParamsDefaultOnly("claude-sonnet-4");

    const select = await screen.findByRole("combobox", { name: "Default model" });
    expect(select).toHaveValue("claude-sonnet-4");

    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig()).toMatchObject({ default_model: "claude-sonnet-4" });
    expect(savedDefaultModel()).toBe("claude-sonnet-4");
  });

  // The config blob marker is this UI's own authoritative record of intent (see
  // hydratePinnedDefaultModel), so it wins even over a litellm_params value that disagrees —
  // e.g. a stale value from before the operator most recently changed the pin.
  it("prefers the config blob marker over a diverging litellm_params value", async () => {
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
            complexity_router_config: { ...STORED_CONFIG, default_model: "blob-pin" },
            complexity_router_default_model: "stale-litellm-params-value",
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

    const select = await screen.findByRole("combobox", { name: "Default model" });
    expect(select).toHaveValue("blob-pin");

    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig()).toMatchObject({ default_model: "blob-pin" });
  });

  // This modal only requires one non-empty tier, so a COMPLEX-only router is reachable here even
  // though the backend raises on it. The block keeps that failure at save time instead of init.
  it("blocks a save when neither the tiers nor a pin give the backend a default", async () => {
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
              tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: ["complex-model"], REASONING: [] },
            },
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

    await user.click(await screen.findByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(toast.fromError).toHaveBeenCalledWith(expect.stringContaining("Simple or Medium tier")));
    expect(modelPatchUpdateCall).not.toHaveBeenCalled();
  });

  it("leaves a router with no stored pin tracking its tiers", async () => {
    const user = userEvent.setup();
    renderWithStoredPin();

    const select = await screen.findByRole("combobox", { name: "Default model" });
    expect(select).toHaveValue("");

    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedDefaultModel()).toBe(STORED_CONFIG.tiers.MEDIUM[0]);
    expect(savedConfig()).not.toHaveProperty("default_model");
  });
});

describe("EditAutoRouterModal plan-mode minimum tier", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  const renderWithStoredTier = (plan_mode_min_tier?: string) =>
    renderWithProviders(
      <EditAutoRouterModal
        isVisible
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
        modelData={{
          ...MODEL_DATA,
          litellm_params: {
            ...MODEL_DATA.litellm_params,
            complexity_router_config: { ...STORED_CONFIG, ...(plan_mode_min_tier && { plan_mode_min_tier }) },
          },
        }}
        accessToken="token"
        userRole="Admin"
      />,
    );

  const openPlanModePanel = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByText("Advanced: Plan-Mode Override"));
  };

  it("shows a stored tier as an enabled override, so the saved value is not a hidden one", async () => {
    const user = userEvent.setup();
    renderWithStoredTier("MEDIUM");
    await openPlanModePanel(user);
    expect(await screen.findByRole("switch", { name: "Route plan-mode requests to a minimum tier" })).toBeChecked();
  });

  it("preserves a stored tier through an untouched open-and-save", async () => {
    const user = userEvent.setup();
    renderWithStoredTier("MEDIUM");

    await user.click(await screen.findByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig()).toMatchObject({ plan_mode_min_tier: "MEDIUM" });
  });

  it("turning the override off removes the stored tier from the saved config", async () => {
    const user = userEvent.setup();
    renderWithStoredTier("MEDIUM");
    await openPlanModePanel(user);
    await user.click(await screen.findByRole("switch", { name: "Route plan-mode requests to a minimum tier" }));

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());
    expect(savedConfig()).not.toHaveProperty("plan_mode_min_tier");
  });
});

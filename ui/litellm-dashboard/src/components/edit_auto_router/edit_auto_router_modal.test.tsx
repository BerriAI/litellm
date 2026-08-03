import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

import NotificationsManager from "@/components/molecules/notifications_manager";
import EditAutoRouterModal from "./edit_auto_router_modal";

const { modelPatchUpdateCall, modelAvailableCall } = vi.hoisted(() => ({
  modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("../networking", () => ({ modelPatchUpdateCall, modelAvailableCall }));

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

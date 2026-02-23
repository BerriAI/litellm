import * as networking from "@/components/networking";
import { fireEvent, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import GuardrailInfoView from "./guardrail_info";

// Mock the networking module
vi.mock("@/components/networking", () => ({
  getGuardrailInfo: vi.fn(),
  getGuardrailUISettings: vi.fn(),
  getGuardrailProviderSpecificParams: vi.fn(),
  updateGuardrailCall: vi.fn(),
}));


// Mock ContentFilterManager
vi.mock("./content_filter/ContentFilterManager", () => ({
  __esModule: true,
  default: ({ onUnsavedChanges, onDataChange, isEditing }: any) => (
    <div data-testid="mock-content-filter-manager">
      {isEditing && (
        <button onClick={() => {
          onUnsavedChanges(true);
          onDataChange?.(["new_pattern"], ["new_word"], []);
        }}>
          Simulate Change
        </button>
      )}
    </div>
  ),
  formatContentFilterDataForAPI: (patterns: any[], blockedWords: any[], categories?: any[]) => ({
    patterns,
    blocked_words: blockedWords,
    categories: categories ?? [],
  }),
}));

describe("Guardrail Info", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the guardrail info after loading", async () => {
    // Mock the network responses
    vi.mocked(networking.getGuardrailInfo).mockResolvedValue({
      guardrail_id: "123",
      guardrail_name: "Test Guardrail",
      litellm_params: {
        guardrail: "presidio",
        mode: "pre_call",
        default_on: true,
      },
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
      guardrail_definition_location: "database",
    });

    vi.mocked(networking.getGuardrailUISettings).mockResolvedValue({
      supported_entities: ["PERSON", "EMAIL"],
      supported_actions: ["MASK", "REDACT"],
      pii_entity_categories: [],
      supported_modes: ["pre_call", "post_call"],
    });

    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue({});

    const { getAllByText, getByText } = render(
      <GuardrailInfoView guardrailId="123" onClose={() => { }} accessToken="123" isAdmin={true} />,
    );

    // Wait for the loading to complete and data to be rendered
    await waitFor(() => {
      // The guardrail name appears in multiple places (title and settings tab)
      const elements = getAllByText("Test Guardrail");
      expect(elements.length).toBeGreaterThan(0);
    });

    // Verify other key elements are present
    expect(getByText("Back to Guardrails")).toBeInTheDocument();
    expect(getByText("Overview")).toBeInTheDocument();
    expect(getByText("Settings")).toBeInTheDocument();
  });

  it("should not render the edit button for config guardrails", async () => {
    // Mock the network responses
    vi.mocked(networking.getGuardrailInfo).mockResolvedValue({
      guardrail_id: "123",
      guardrail_name: "Test Guardrail",
      litellm_params: {
        guardrail: "presidio",
        mode: "pre_call",
        default_on: true,
      },
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
      guardrail_definition_location: "config",
    });

    vi.mocked(networking.getGuardrailUISettings).mockResolvedValue({
      supported_entities: ["PERSON", "EMAIL"],
      supported_actions: ["MASK", "REDACT"],
      pii_entity_categories: [],
      supported_modes: ["pre_call", "post_call"],
    });

    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue({});

    const { getByText, container } = render(
      <GuardrailInfoView guardrailId="123" onClose={() => { }} accessToken="123" isAdmin={true} />,
    );

    await waitFor(() => {
      expect(getByText("Settings")).toBeInTheDocument();
    });

    // Click the Settings tab
    fireEvent.click(getByText("Settings"));

    // Wait for the Settings panel to render
    await waitFor(() => {
      expect(getByText("Guardrail Settings")).toBeInTheDocument();
    });

    // Find the info icon and hover over it
    const infoIcon = container.querySelector(".anticon-info-circle");
    expect(infoIcon).toBeInTheDocument();

    if (infoIcon) {
      fireEvent.mouseEnter(infoIcon);

      // Wait for the tooltip to appear
      await waitFor(() => {
        expect(getByText("Guardrail is defined in the config file and cannot be edited.")).toBeInTheDocument();
      });
    }
  });

  it("should render the guardrail info", async () => {
    // Mock the network responses
    vi.mocked(networking.getGuardrailInfo).mockResolvedValue({
      guardrail_id: "123",
      guardrail_name: "Test Guardrail",
      litellm_params: {
        guardrail: "presidio",
        mode: "pre_call",
        default_on: true,
        pii_entities_config: {
          PERSON: "MASK",
          EMAIL: "REDACT",
        },
      },
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
      guardrail_definition_location: "database",
    });

    vi.mocked(networking.getGuardrailUISettings).mockResolvedValue({
      supported_entities: ["PERSON", "EMAIL"],
      supported_actions: ["MASK", "REDACT"],
      pii_entity_categories: [],
      supported_modes: ["pre_call", "post_call"],
    });

    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue({});

    const { getByText } = render(
      <GuardrailInfoView guardrailId="123" onClose={() => { }} accessToken="123" isAdmin={true} />,
    );

    await waitFor(() => {
      expect(getByText("PII Entity Configuration")).toBeInTheDocument();
    });
  });
  it("should handle content filter updates correctly", async () => {
    // Mock the network responses
    vi.mocked(networking.getGuardrailInfo).mockResolvedValue({
      guardrail_id: "123",
      guardrail_name: "Content Filter Guardrail",
      litellm_params: {
        guardrail: "litellm_content_filter",
        mode: "pre_call",
        default_on: true,
        patterns: ["initial_pattern"],
        blocked_words: ["initial_word"],
      },
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
      guardrail_definition_location: "database",
    });

    vi.mocked(networking.getGuardrailUISettings).mockResolvedValue({
      supported_entities: [],
      supported_actions: [],
      pii_entity_categories: [],
      supported_modes: ["pre_call", "post_call"],
    });

    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue({});
    vi.mocked(networking.updateGuardrailCall).mockResolvedValue({ status: "success" });

    const { getByText, getByRole, getAllByRole, getByLabelText } = render(
      <GuardrailInfoView guardrailId="123" onClose={() => { }} accessToken="123" isAdmin={true} />,
    );

    await waitFor(() => {
      expect(getByText("Settings")).toBeInTheDocument();
    });

    // Go to Settings tab
    fireEvent.click(getByText("Settings"));

    await waitFor(() => {
      expect(getByText("Guardrail Settings")).toBeInTheDocument();
    });

    // Enter Edit Mode
    fireEvent.click(getByText("Edit Settings"));

    // Modify Guardrail Name to force an update
    const nameInput = getByLabelText("Guardrail Name");
    fireEvent.change(nameInput, { target: { value: "Updated Name" } });

    // Save with only name change
    const saveButton = getByText("Save Changes");
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(networking.updateGuardrailCall).toHaveBeenCalled();
    });

    // Verify call did NOT include patterns or blocked_words (because no changes)
    // updateGuardrailCall(accessToken, guardrailId, updateData) -> index 2 is updateData
    const firstCallArgs: any = vi.mocked(networking.updateGuardrailCall).mock.calls[0][2];

    // Verify attributes that definitely changed
    expect(firstCallArgs.guardrail_name).toBe("Updated Name");

    // litellm_params might be undefined if empty, which is correct. 
    // If it exists, ensure patterns/blocked_words are not in it.
    if (firstCallArgs.litellm_params) {
      expect(firstCallArgs.litellm_params.patterns).toBeUndefined();
      expect(firstCallArgs.litellm_params.blocked_words).toBeUndefined();
    }

    // Clear mocks to reset call count
    vi.clearAllMocks();

    // Enter Edit Mode again to make changes
    await waitFor(() => {
      expect(getByText("Edit Settings")).toBeInTheDocument();
    });
    fireEvent.click(getByText("Edit Settings"));

    // Now modify the values using the mock button
    const simulateChangeButton = getByText("Simulate Change");
    fireEvent.click(simulateChangeButton);

    // Save again
    fireEvent.click(getByText("Save Changes"));

    await waitFor(() => {
      expect(networking.updateGuardrailCall).toHaveBeenCalled();
    });

    // Verify call INCLUDES patterns and blocked_words
    const secondCallArgs: any = vi.mocked(networking.updateGuardrailCall).mock.calls[0][2];
    expect(secondCallArgs.litellm_params).toBeDefined();
    expect(secondCallArgs.litellm_params.patterns).toEqual(["new_pattern"]);
    expect(secondCallArgs.litellm_params.blocked_words).toEqual(["new_word"]);
  });
});

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
      <GuardrailInfoView guardrailId="123" onClose={() => {}} accessToken="123" isAdmin={true} />,
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
      <GuardrailInfoView guardrailId="123" onClose={() => {}} accessToken="123" isAdmin={true} />,
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
      <GuardrailInfoView guardrailId="123" onClose={() => {}} accessToken="123" isAdmin={true} />,
    );

    await waitFor(() => {
      expect(getByText("PII Entity Configuration")).toBeInTheDocument();
    });
  });
});

import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../../../tests/test-utils";
import {
  makeBedrockResponse,
  makeEntity,
  makeGuardrailInformation,
} from "@/components/view_logs/GuardrailViewer/__tests__/fixtures";
import GuardrailViewer from "@/components/view_logs/GuardrailViewer/GuardrailViewer";

// We will mock child components selectively for some tests to assert prop passthrough,
// but also run an integration-style render without mocks.
const PresidioPath = "@/components/view_logs/GuardrailViewer/PresidioDetectedEntities";
const BedrockPath = "@/components/view_logs/GuardrailViewer/BedrockGuardrailDetails";

describe("GuardrailViewer", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("shows header, status pill, and duration", () => {
    const data = makeGuardrailInformation({ duration: 1.23456, guardrail_status: "success" });
    renderWithProviders(<GuardrailViewer data={data} />);

    expect(screen.getByText("Guardrails & Policy Compliance")).toBeInTheDocument();
    // header shows passed count
    expect(screen.getByText(/1 Passed/)).toBeInTheDocument();
    // The PASSED badge in the evaluation card
    expect(screen.getByText("PASSED")).toBeInTheDocument();

    // duration displays in ms format: Math.round(1.23456 * 1000) = 1235
    expect(screen.getByText("1235ms")).toBeInTheDocument();
  });

  it("calculates and displays masked entity totals", async () => {
    const user = userEvent.setup();
    const data = makeGuardrailInformation({
      masked_entity_count: { EMAIL_ADDRESS: 2, PHONE_NUMBER: 1 },
    });
    renderWithProviders(<GuardrailViewer data={data} />);

    // In collapsed state, the match count badge is visible
    expect(screen.getByText("3 matched")).toBeInTheDocument();

    // Expand the evaluation card to see entity details
    await user.click(screen.getByText("pii-rail"));
    // summary chips for each entry inside expanded card
    expect(screen.getByText("EMAIL_ADDRESS: 2")).toBeInTheDocument();
    expect(screen.getByText("PHONE_NUMBER: 1")).toBeInTheDocument();
  });

  it("hides matched badge when count is zero/empty", () => {
    const data = makeGuardrailInformation({ masked_entity_count: {} });
    renderWithProviders(<GuardrailViewer data={data} />);

    expect(screen.queryByText(/matched/)).not.toBeInTheDocument();
  });

  it("toggles evaluation card open/closed on click", async () => {
    const user = userEvent.setup();
    const data = makeGuardrailInformation({
      masked_entity_count: { EMAIL_ADDRESS: 2 },
    });
    renderWithProviders(<GuardrailViewer data={data} />);

    // Initially collapsed — masked entity details not visible
    expect(screen.queryByText("EMAIL_ADDRESS: 2")).not.toBeInTheDocument();

    // Click to expand
    await user.click(screen.getByText("pii-rail"));
    expect(screen.getByText("EMAIL_ADDRESS: 2")).toBeInTheDocument();

    // Click again to collapse
    await user.click(screen.getByText("pii-rail"));
    await waitFor(() => {
      expect(screen.queryByText("EMAIL_ADDRESS: 2")).not.toBeInTheDocument();
    });
  });

  it("defaults to presidio provider when guardrail_provider is undefined", async () => {
    vi.doMock(PresidioPath, () => ({
      __esModule: true,
      default: ({ entities }: any) => <div data-testid="presidio-mock">presidio {entities?.length}</div>,
    }));
    const { default: Component } = await import("@/components/view_logs/GuardrailViewer/GuardrailViewer");

    const data = makeGuardrailInformation({
      guardrail_provider: undefined,
      guardrail_response: [makeEntity(), makeEntity()],
    });
    renderWithProviders(<Component data={data} />);

    // Expand the card to see provider-specific content
    const user = userEvent.setup();
    await user.click(screen.getByText("pii-rail"));
    expect(screen.getByTestId("presidio-mock")).toHaveTextContent("presidio 2");
  });

  it('renders PresidioDetectedEntities when provider="presidio" and response has entities', async () => {
    vi.doMock(PresidioPath, () => ({
      __esModule: true,
      default: ({ entities }: any) => <div data-testid="presidio-mock">count:{entities?.length}</div>,
    }));
    const { default: Component } = await import("@/components/view_logs/GuardrailViewer/GuardrailViewer");

    const data = makeGuardrailInformation({
      guardrail_provider: "presidio",
      guardrail_response: [makeEntity()],
    });
    renderWithProviders(<Component data={data} />);

    // Expand the card to see provider-specific content
    const user = userEvent.setup();
    await user.click(screen.getByText("pii-rail"));
    expect(screen.getByTestId("presidio-mock")).toHaveTextContent("count:1");
  });

  it('renders BedrockGuardrailDetails when provider="bedrock"', async () => {
    vi.doMock(BedrockPath, () => ({
      __esModule: true,
      default: ({ response }: any) => <div data-testid="bedrock-mock">{response?.action ?? "no-action"}</div>,
    }));
    const { default: Component } = await import("@/components/view_logs/GuardrailViewer/GuardrailViewer");

    const data = makeGuardrailInformation({
      guardrail_provider: "bedrock",
      guardrail_response: makeBedrockResponse({ action: "GUARDRAIL_INTERVENED" }),
    });
    renderWithProviders(<Component data={data} />);

    // Expand the card to see provider-specific content
    const user = userEvent.setup();
    await user.click(screen.getByText("pii-rail"));
    expect(screen.getByTestId("bedrock-mock")).toHaveTextContent("GUARDRAIL_INTERVENED");
  });

  it("unknown provider renders neither Presidio nor Bedrock details", async () => {
    const user = userEvent.setup();
    const data = makeGuardrailInformation({
      guardrail_provider: "unknown",
    });
    renderWithProviders(<GuardrailViewer data={data} />);
    // Header still present
    expect(screen.getByText("Guardrails & Policy Compliance")).toBeInTheDocument();

    // Expand the card
    await user.click(screen.getByText("pii-rail"));
    // No Presidio or Bedrock sections
    expect(screen.queryByText(/Detected Entities/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Raw Bedrock Guardrail Response/)).not.toBeInTheDocument();
  });

  it("integration: renders with real Bedrock details without mocks", async () => {
    const user = userEvent.setup();
    const data = makeGuardrailInformation({
      guardrail_provider: "bedrock",
      guardrail_response: makeBedrockResponse({
        action: "NONE",
        outputs: [{ text: "ok" }],
      }),
    });
    renderWithProviders(<GuardrailViewer data={data} />);

    // Expand the card to reveal Bedrock details
    await user.click(screen.getByText("pii-rail"));

    // Bedrock summary bits
    expect(screen.getByText("Outputs")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });
});

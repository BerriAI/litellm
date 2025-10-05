import React from "react";
import { describe, it, expect } from "vitest";
import userEvent from "@testing-library/user-event";
import PresidioDetectedEntities from "@/components/view_logs/GuardrailViewer/PresidioDetectedEntities";
import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { makeEntity } from "@/components/view_logs/GuardrailViewer/__tests__/fixtures";

describe("PresidioDetectedEntities", () => {
  it("renders null when entities empty", () => {
    const { container } = renderWithProviders(<PresidioDetectedEntities entities={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders per-entity header info including score color and position", async () => {
    const user = userEvent.setup();
    const e = makeEntity({ start: 10, end: 20, score: 0.92, entity_type: "EMAIL_ADDRESS" });
    renderWithProviders(<PresidioDetectedEntities entities={[e]} />);

    // Header row values
    expect(screen.getByText("EMAIL_ADDRESS")).toBeInTheDocument();
    expect(screen.getByText(/Score: 0\.92/)).toBeInTheDocument();
    expect(screen.getByText("Position: 10-20")).toBeInTheDocument();

    // Expand details
    await user.click(screen.getByText("EMAIL_ADDRESS"));
    expect(screen.getByText("Entity Type:")).toBeInTheDocument();
    expect(screen.getByText("Characters 10-20")).toBeInTheDocument();
    expect(screen.getByText("Confidence:")).toBeInTheDocument();
    // Recognizer details
    expect(screen.getByText("EmailRecognizer")).toBeInTheDocument();
    expect(screen.getByText("email_v1")).toBeInTheDocument();
    // Explanation
    expect(screen.getByText("Matched via pattern")).toBeInTheDocument();
  });

  it("handles missing metadata & low scores gracefully", async () => {
    const user = userEvent.setup();
    const e = makeEntity({
      score: 0.3,
      recognition_metadata: undefined as any,
      analysis_explanation: null,
      entity_type: "NAME",
      start: 0,
      end: 0,
    });
    renderWithProviders(<PresidioDetectedEntities entities={[e]} />);

    await user.click(screen.getByText("NAME"));
    // No recognizer/explanation rows
    expect(screen.queryByText("Recognizer:")).not.toBeInTheDocument();
    expect(screen.queryByText("Explanation:")).not.toBeInTheDocument();
    // Position still renders
    expect(screen.getByText("Characters 0-0")).toBeInTheDocument();
  });
});

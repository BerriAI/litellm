import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AddFallbacksModal } from "./AddFallbacksModal";

describe("AddFallbacksModal", () => {
  const mockOnCancel = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the modal when open is true", () => {
    render(
      <AddFallbacksModal open={true} onCancel={mockOnCancel}>
        <div>Test Content</div>
      </AddFallbacksModal>,
    );

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Configure Model Fallbacks")).toBeInTheDocument();
    expect(screen.getByText("Manage multiple fallback chains for different models (up to 5 groups at a time)")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should not render the modal when open is false", () => {
    render(
      <AddFallbacksModal open={false} onCancel={mockOnCancel}>
        <div>Test Content</div>
      </AddFallbacksModal>,
    );

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("should render children content when modal is open", () => {
    render(
      <AddFallbacksModal open={true} onCancel={mockOnCancel}>
        <div data-testid="child-content">Child Component</div>
      </AddFallbacksModal>,
    );

    expect(screen.getByTestId("child-content")).toBeInTheDocument();
    expect(screen.getByText("Child Component")).toBeInTheDocument();
  });

  it("should display the correct title and description", () => {
    render(
      <AddFallbacksModal open={true} onCancel={mockOnCancel}>
        <div>Content</div>
      </AddFallbacksModal>,
    );

    expect(screen.getByText("Configure Model Fallbacks")).toBeInTheDocument();
    expect(screen.getByText(/Manage multiple fallback chains/i)).toBeInTheDocument();
  });
});

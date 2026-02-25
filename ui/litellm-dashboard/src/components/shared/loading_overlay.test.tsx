import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LoadingOverlay } from "./loading_overlay";

describe("LoadingOverlay", () => {
  it("should render children when not loading", () => {
    render(
      <LoadingOverlay loading={false}>
        <div data-testid="child-content">Content</div>
      </LoadingOverlay>
    );
    expect(screen.getByTestId("child-content")).toBeInTheDocument();
    expect(screen.getByText("Content")).toBeInTheDocument();
    expect(screen.queryByTestId("loading-overlay")).not.toBeInTheDocument();
  });

  it("should render loading overlay when loading is true", () => {
    render(
      <LoadingOverlay loading={true}>
        <div data-testid="child-content">Content</div>
      </LoadingOverlay>
    );
    expect(screen.getByTestId("child-content")).toBeInTheDocument();
    expect(screen.getByText("Content")).toBeInTheDocument();
    expect(screen.getByTestId("loading-overlay")).toBeInTheDocument();
  });

  it("should render optional message when provided and loading", () => {
    render(
      <LoadingOverlay loading={true} message="Fetching data...">
        <div>Content</div>
      </LoadingOverlay>
    );
    expect(screen.getByText("Fetching data...")).toBeInTheDocument();
  });

  it("should use overlay variant by default (blurred content, centered loader)", () => {
    const { container } = render(
      <LoadingOverlay loading={true} message="Updating data...">
        <div data-testid="child-content">Old data</div>
      </LoadingOverlay>
    );
    const overlay = screen.getByTestId("loading-overlay");
    expect(overlay).toHaveClass("inset-0");
    const contentWrapper = container.querySelector(".blur-\\[2px\\]");
    expect(contentWrapper).toBeInTheDocument();
    expect(screen.getByText("Old data")).toBeInTheDocument();
  });

  it("should use subtle variant when specified (content visible, corner badge)", () => {
    const { container } = render(
      <LoadingOverlay loading={true} variant="subtle" message="Updating data...">
        <div data-testid="child-content">Old data</div>
      </LoadingOverlay>
    );
    const overlay = screen.getByTestId("loading-overlay");
    expect(overlay).toHaveClass("top-3", "right-3");
    expect(overlay).not.toHaveClass("inset-0");
  });

  it("should blur content when variant is overlay", () => {
    const { container } = render(
      <LoadingOverlay loading={true} variant="overlay">
        <div data-testid="child-content">Content</div>
      </LoadingOverlay>
    );
    const contentWrapper = container.querySelector(".blur-\\[2px\\]");
    expect(contentWrapper).toBeInTheDocument();
  });
});

import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./UsageTab", () => ({ __esModule: true, default: () => <div data-testid="usage-tab" /> }));
vi.mock("./PromptCompressionTab", () => ({ __esModule: true, default: () => <div data-testid="compression-tab" /> }));
vi.mock("./AutorouterTab", () => ({ __esModule: true, default: () => <div data-testid="autorouter-tab" /> }));
vi.mock("./PromptCachingTab", () => ({ __esModule: true, default: () => <div data-testid="caching-tab" /> }));

import CostOptimizationView from "./CostOptimizationView";

const renderView = () => render(<CostOptimizationView accessToken="test-token" userId="u1" userRole="proxy_admin" />);

describe("CostOptimizationView", () => {
  it("renders all four cost-optimization tabs", () => {
    const { getByText } = renderView();

    expect(getByText("Usage")).toBeInTheDocument();
    expect(getByText("Prompt Compression")).toBeInTheDocument();
    expect(getByText("Autorouter")).toBeInTheDocument();
    expect(getByText("Prompt Caching")).toBeInTheDocument();
  });

  it("defaults to the Usage tab and switches the active tab on click", () => {
    const { getByRole } = renderView();

    expect(getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "false");

    fireEvent.click(getByRole("tab", { name: "Prompt Compression" }));

    expect(getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "false");
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "true");
  });
});

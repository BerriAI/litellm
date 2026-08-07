import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./UsageTab", () => ({ __esModule: true, default: () => <div data-testid="usage-tab" /> }));
vi.mock("./PromptCompressionTab", () => ({ __esModule: true, default: () => <div data-testid="compression-tab" /> }));
vi.mock("./PromptCachingTab", () => ({ __esModule: true, default: () => <div data-testid="caching-tab" /> }));
vi.mock("./AutoRouterBenchmarksTab", () => ({
  __esModule: true,
  default: () => <div data-testid="autorouter-benchmarks-tab" />,
}));

import CostOptimizationView from "./CostOptimizationView";

const renderView = () => render(<CostOptimizationView accessToken="test-token" userId="u1" userRole="proxy_admin" />);

describe("CostOptimizationView", () => {
  it("renders the four cost-optimization tabs", () => {
    const { getByText } = renderView();

    expect(getByText("Overall")).toBeInTheDocument();
    expect(getByText("Prompt Compression")).toBeInTheDocument();
    expect(getByText("Prompt Caching")).toBeInTheDocument();
    expect(getByText("Auto-Router")).toBeInTheDocument();
  });

  it("defaults to the Overall tab and switches the active tab on click", () => {
    const { getByRole } = renderView();

    expect(getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "false");

    fireEvent.click(getByRole("tab", { name: "Prompt Compression" }));

    expect(getByRole("tab", { name: "Overall" })).toHaveAttribute("aria-selected", "false");
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "true");
  });
});

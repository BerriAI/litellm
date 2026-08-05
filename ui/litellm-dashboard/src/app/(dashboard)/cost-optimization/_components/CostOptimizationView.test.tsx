import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./UsageTab", () => ({ __esModule: true, default: () => <div data-testid="usage-tab" /> }));
vi.mock("./PromptCachingTab", () => ({ __esModule: true, default: () => <div data-testid="caching-tab" /> }));

import CostOptimizationView from "./CostOptimizationView";

const renderView = () => render(<CostOptimizationView accessToken="test-token" userId="u1" userRole="proxy_admin" />);

describe("CostOptimizationView", () => {
  it("renders the reporting tabs, with compression config moved to its own page", () => {
    const { getByText, queryByText } = renderView();

    expect(getByText("Usage")).toBeInTheDocument();
    expect(getByText("Prompt Caching")).toBeInTheDocument();
    expect(queryByText("Prompt Compression")).not.toBeInTheDocument();
    expect(queryByText("Autorouter")).not.toBeInTheDocument();
  });

  it("defaults to the Usage tab and switches the active tab on click", () => {
    const { getByRole } = renderView();

    expect(getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Prompt Caching" })).toHaveAttribute("aria-selected", "false");

    fireEvent.click(getByRole("tab", { name: "Prompt Caching" }));

    expect(getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "false");
    expect(getByRole("tab", { name: "Prompt Caching" })).toHaveAttribute("aria-selected", "true");
  });
});

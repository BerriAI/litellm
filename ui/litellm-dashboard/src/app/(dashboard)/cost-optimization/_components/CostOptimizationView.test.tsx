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

  it("defaults to the Usage tab and switches to a config tab on click", () => {
    const { getByText, getByTestId } = renderView();

    expect(getByTestId("usage-tab")).toBeInTheDocument();

    fireEvent.click(getByText("Prompt Compression"));
    expect(getByTestId("compression-tab")).toBeInTheDocument();
  });
});

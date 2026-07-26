import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { AutoRouterModelGroupsProvider, AutoRouterTag } from "./AutoRouterTag";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAutoRouterModelGroups: vi.fn(),
}));

import { useAutoRouterModelGroups } from "@/app/(dashboard)/hooks/models/useModels";

const mockUseAutoRouterModelGroups = vi.mocked(useAutoRouterModelGroups);

const renderInProvider = (ui: React.ReactNode) =>
  render(<AutoRouterModelGroupsProvider>{ui}</AutoRouterModelGroupsProvider>);

describe("AutoRouterTag", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("names the router that served the request", () => {
    mockUseAutoRouterModelGroups.mockReturnValue(new Set(["smart-router"]));

    renderInProvider(<AutoRouterTag modelGroup="smart-router" />);

    const tag = screen.getByTitle('Routed by auto-router "smart-router"');
    expect(tag).toHaveTextContent("smart-router");
    expect(tag.querySelector("svg")).not.toBeNull();
  });

  it("does not tag a plain alias whose model group differs from the resolved model", () => {
    mockUseAutoRouterModelGroups.mockReturnValue(new Set(["smart-router"]));

    renderInProvider(<AutoRouterTag modelGroup="claude-haiku" />);

    expect(screen.queryByText("claude-haiku")).not.toBeInTheDocument();
  });

  it("renders nothing while the model list is unavailable, so a routed row is never mislabelled", () => {
    mockUseAutoRouterModelGroups.mockReturnValue(new Set<string>());

    const { container } = renderInProvider(<AutoRouterTag modelGroup="smart-router" />);

    expect(container).toBeEmptyDOMElement();
  });

  it.each([[undefined], [null], [""]])("renders nothing when the row carries no model group (%s)", (modelGroup) => {
    mockUseAutoRouterModelGroups.mockReturnValue(new Set(["smart-router"]));

    const { container } = renderInProvider(<AutoRouterTag modelGroup={modelGroup} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing outside a provider instead of requiring a QueryClient", () => {
    const { container } = render(<AutoRouterTag modelGroup="smart-router" />);

    expect(container).toBeEmptyDOMElement();
    expect(mockUseAutoRouterModelGroups).not.toHaveBeenCalled();
  });
});

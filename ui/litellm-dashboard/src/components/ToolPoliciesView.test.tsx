import React from "react";
import { beforeEach, describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../tests/test-utils";
import ToolPoliciesView from "./ToolPoliciesView";

let currentUrl = "/ui/tool-policies";
const urlListeners = new Set<() => void>();

vi.mock("next/navigation", async () => {
  const react = await import("react");
  return {
    useRouter: () => ({
      push: (url: string) => {
        currentUrl = url;
        urlListeners.forEach((listener) => listener());
      },
    }),
    usePathname: () => currentUrl.split("?")[0],
    useSearchParams: () => {
      const [, rerender] = react.useReducer((n: number) => n + 1, 0);
      react.useEffect(() => {
        urlListeners.add(rerender);
        return () => {
          urlListeners.delete(rerender);
        };
      }, [rerender]);
      return new URLSearchParams(currentUrl.split("?")[1] ?? "");
    },
  };
});

vi.mock("@/components/ToolDetail", () => ({
  ToolDetail: ({ toolName, onBack }: { toolName: string; onBack: () => void }) => (
    <div>
      <span>Detail: {toolName}</span>
      <button onClick={onBack}>Back</button>
    </div>
  ),
}));

vi.mock("@/components/ToolPolicies/ToolPoliciesPanel", () => ({
  ToolPoliciesPanel: function ToolPoliciesPanelMock({ onSelectTool }: { onSelectTool: (name: string) => void }) {
    return (
      <div>
        <span>Tool Policies Overview</span>
        <button onClick={() => onSelectTool("my-tool")}>Select Tool</button>
      </div>
    );
  },
}));

describe("ToolPoliciesView", () => {
  beforeEach(() => {
    currentUrl = "/ui/tool-policies";
  });

  it("should render the overview by default", () => {
    renderWithProviders(<ToolPoliciesView accessToken="token" />);

    expect(screen.getByText("Tool Policies Overview")).toBeInTheDocument();
  });

  it("should navigate to tool detail when a tool is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ToolPoliciesView accessToken="token" />);

    await user.click(screen.getByRole("button", { name: /select tool/i }));

    expect(screen.getByText("Detail: my-tool")).toBeInTheDocument();
    expect(screen.queryByText("Tool Policies Overview")).not.toBeInTheDocument();
    expect(currentUrl).toBe("/ui/tool-policies?tool=my-tool");
  });

  it("should open a tool's detail directly from the ?tool= deep link", () => {
    currentUrl = "/ui/tool-policies?tool=deep-linked-tool";
    renderWithProviders(<ToolPoliciesView accessToken="token" />);

    expect(screen.getByText("Detail: deep-linked-tool")).toBeInTheDocument();
  });

  it("should navigate back to overview when back is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ToolPoliciesView accessToken="token" />);

    await user.click(screen.getByRole("button", { name: /select tool/i }));
    await user.click(screen.getByRole("button", { name: /back/i }));

    expect(screen.getByText("Tool Policies Overview")).toBeInTheDocument();
    expect(screen.queryByText("Detail: my-tool")).not.toBeInTheDocument();
  });
});

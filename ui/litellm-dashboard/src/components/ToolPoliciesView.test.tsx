import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../tests/test-utils";
import ToolPoliciesView from "./ToolPoliciesView";

vi.mock("@/components/ToolDetail", () => ({
  ToolDetail: ({ toolName, onBack }: { toolName: string; onBack: () => void }) => (
    <div>
      <span>Detail: {toolName}</span>
      <button onClick={onBack}>Back</button>
    </div>
  ),
}));

vi.mock("@/components/ToolPolicies", () => ({
  ToolPolicies: ({ onSelectTool }: { onSelectTool: (name: string) => void }) => (
    <div>
      <span>Tool Policies Overview</span>
      <button onClick={() => onSelectTool("my-tool")}>Select Tool</button>
    </div>
  ),
}));

describe("ToolPoliciesView", () => {
  it("should render the overview by default", () => {
    renderWithProviders(<ToolPoliciesView accessToken="token" userRole="Admin" />);

    expect(screen.getByText("Tool Policies Overview")).toBeInTheDocument();
  });

  it("should navigate to tool detail when a tool is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ToolPoliciesView accessToken="token" userRole="Admin" />);

    await user.click(screen.getByRole("button", { name: /select tool/i }));

    expect(screen.getByText("Detail: my-tool")).toBeInTheDocument();
    expect(screen.queryByText("Tool Policies Overview")).not.toBeInTheDocument();
  });

  it("should navigate back to overview when back is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ToolPoliciesView accessToken="token" userRole="Admin" />);

    await user.click(screen.getByRole("button", { name: /select tool/i }));
    await user.click(screen.getByRole("button", { name: /back/i }));

    expect(screen.getByText("Tool Policies Overview")).toBeInTheDocument();
    expect(screen.queryByText("Detail: my-tool")).not.toBeInTheDocument();
  });
});

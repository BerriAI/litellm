import React from "react";
import { beforeEach, describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { renderWithProviders } from "../../tests/test-utils";
import ToolPoliciesView from "./ToolPoliciesView";

const can = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: (...args: unknown[]) => can(...args),
}));

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
    can.mockReset().mockReturnValue(true);
  });

  it("should show an admin-only notice instead of the overview when the caller lacks access", () => {
    can.mockReturnValue(false);
    renderWithProviders(<ToolPoliciesView accessToken="token" />);

    expect(screen.getByText(/only available to admin users/i)).toBeInTheDocument();
    expect(screen.queryByText("Tool Policies Overview")).not.toBeInTheDocument();
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
  });

  it("should navigate back to overview when back is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ToolPoliciesView accessToken="token" />);

    await user.click(screen.getByRole("button", { name: /select tool/i }));
    await user.click(screen.getByRole("button", { name: /back/i }));

    expect(screen.getByText("Tool Policies Overview")).toBeInTheDocument();
    expect(screen.queryByText("Detail: my-tool")).not.toBeInTheDocument();
  });

  describe("?tool= deep link", () => {
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];

    it("should open the detail for the tool named in the URL", () => {
      renderWithProviders(<ToolPoliciesView accessToken="token" />, { searchParams: "?tool=get_weather" });

      expect(screen.getByText("Detail: get_weather")).toBeInTheDocument();
      expect(screen.queryByText("Tool Policies Overview")).not.toBeInTheDocument();
    });

    it("should show the overview when the tool param is empty", () => {
      renderWithProviders(<ToolPoliciesView accessToken="token" />, { searchParams: "?tool=" });

      expect(screen.getByText("Tool Policies Overview")).toBeInTheDocument();
    });

    it("should push the selected tool to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<ToolPoliciesView accessToken="token" />, { onUrlUpdate });

      await user.click(screen.getByRole("button", { name: /select tool/i }));

      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tool")).toBe("my-tool");
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    });

    it("should clear the tool from the URL when going back", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<ToolPoliciesView accessToken="token" />, { searchParams: "?tool=get_weather", onUrlUpdate });

      await user.click(screen.getByRole("button", { name: /back/i }));

      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tool")).toBe(false);
      expect(await screen.findByText("Tool Policies Overview")).toBeInTheDocument();
    });

    it("should keep the admin-only notice for a caller without access even with a tool in the URL", () => {
      can.mockReturnValue(false);
      renderWithProviders(<ToolPoliciesView accessToken="token" />, { searchParams: "?tool=get_weather" });

      expect(screen.getByText(/only available to admin users/i)).toBeInTheDocument();
      expect(screen.queryByText("Detail: get_weather")).not.toBeInTheDocument();
    });
  });
});

import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { focusManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import type { ToolRow } from "@/components/networking";
import { ToolPoliciesPanel } from "./ToolPoliciesPanel";

const fetchToolsList = vi.fn();
const updateToolPolicy = vi.fn();

vi.mock("@/components/networking", () => ({
  fetchToolsList: (...args: unknown[]) => fetchToolsList(...args),
  updateToolPolicy: (...args: unknown[]) => updateToolPolicy(...args),
}));

const fromBackend = vi.fn();
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { fromBackend: (...args: unknown[]) => fromBackend(...args) },
}));

const NOW = new Date("2026-07-21T12:00:00Z");

const TOOLS: ToolRow[] = [
  {
    tool_id: "tool-1",
    tool_name: "get_weather",
    input_policy: "untrusted",
    output_policy: "untrusted",
    call_count: 12,
    team_id: "team-alpha",
    key_alias: "prod-key",
    key_hash: "hash-aaa",
    user_agent: "curl/8.7.1",
    created_at: "2026-07-21T10:00:00Z",
  },
  {
    tool_id: "tool-2",
    tool_name: "search_web",
    input_policy: "trusted",
    output_policy: "trusted",
    call_count: 5,
    team_id: "team-beta",
    key_alias: "dev-key",
    key_hash: "hash-bbb",
    created_at: "2026-07-20T10:00:00Z",
  },
  {
    tool_id: "tool-3",
    tool_name: "delete_file",
    input_policy: "blocked",
    output_policy: "untrusted",
    call_count: 100,
    key_hash: "hash-ccc",
    created_at: "2026-07-19T10:00:00Z",
  },
];

const row = (toolId: string): HTMLElement => {
  const element = document.querySelector(`[data-row-id="${toolId}"]`);
  if (element === null) throw new Error(`row ${toolId} is not rendered`);
  return element as HTMLElement;
};

const policySelect = (toolId: string, kind: "input" | "output"): HTMLElement =>
  within(row(toolId)).getAllByRole("combobox")[kind === "input" ? 0 : 1];

const chooseOption = async (user: ReturnType<typeof userEvent.setup>, trigger: HTMLElement, label: string) => {
  await user.click(trigger);
  const option = await waitFor(() => {
    const match = Array.from(document.querySelectorAll(".ant-select-item-option")).find(
      (element) => element.textContent === label,
    );
    if (match === undefined) throw new Error(`option ${label} not open`);
    return match as HTMLElement;
  });
  await user.click(option);
};

const renderPanel = (onSelectTool = vi.fn()) =>
  renderWithProviders(<ToolPoliciesPanel accessToken="sk-token" onSelectTool={onSelectTool} />);

const waitForRows = () => waitFor(() => expect(document.querySelector('[data-row-id="tool-1"]')).not.toBeNull());

beforeEach(() => {
  testQueryClient.clear();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  fetchToolsList.mockReset().mockResolvedValue(TOOLS);
  updateToolPolicy.mockReset().mockResolvedValue({});
  fromBackend.mockReset();
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("ToolPoliciesPanel data loading", () => {
  it("should load tools once and never auto-refresh on a timer", async () => {
    renderPanel();
    await waitForRows();

    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });

    expect(fetchToolsList).toHaveBeenCalledTimes(1);
  });

  it("should not refetch when the window regains focus", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <ToolPoliciesPanel accessToken="sk-token" onSelectTool={vi.fn()} />
      </QueryClientProvider>,
    );
    await waitForRows();

    await act(async () => {
      focusManager.setFocused(false);
      focusManager.setFocused(true);
    });

    expect(fetchToolsList).toHaveBeenCalledTimes(1);
    focusManager.setFocused(undefined);
  });

  it("should refetch when the toolbar refresh action is used", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPanel();
    await waitForRows();

    await user.click(screen.getByTestId("datatable-refresh"));

    await waitFor(() => expect(fetchToolsList).toHaveBeenCalledTimes(2));
  });

  it("should keep rows visible during a refresh instead of falling back to skeletons", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPanel();
    await waitForRows();

    fetchToolsList.mockReturnValue(new Promise(() => {}));
    await user.click(screen.getByTestId("datatable-refresh"));

    expect(row("tool-1")).toBeInTheDocument();
    expect(screen.queryAllByTestId("skeleton-row")).toHaveLength(0);
  });

  it("should resolve the loading skeleton when there is no access token", async () => {
    renderWithProviders(<ToolPoliciesPanel accessToken={null} onSelectTool={vi.fn()} />);

    await waitFor(() => expect(screen.queryAllByTestId("skeleton-row")).toHaveLength(0));
    expect(fetchToolsList).not.toHaveBeenCalled();
    expect(screen.getByText("No tools discovered")).toBeInTheDocument();
  });

  it("should surface a load failure without wedging the skeleton", async () => {
    fetchToolsList.mockRejectedValue(new Error("boom"));
    renderPanel();

    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
    expect(screen.queryAllByTestId("skeleton-row")).toHaveLength(0);
  });
});

describe("ToolPoliciesPanel inline policy editing", () => {
  it("should patch the input policy and update that row in place", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPanel();
    await waitForRows();

    await chooseOption(user, policySelect("tool-1", "input"), "trusted");

    expect(updateToolPolicy).toHaveBeenCalledWith("sk-token", "get_weather", { input_policy: "trusted" });
    await waitFor(() => expect(within(row("tool-1")).getByText("trusted")).toBeInTheDocument());
    expect(fetchToolsList).toHaveBeenCalledTimes(1);
  });

  it("should patch the output policy from the output column", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPanel();
    await waitForRows();

    await chooseOption(user, policySelect("tool-1", "output"), "trusted");

    expect(updateToolPolicy).toHaveBeenCalledWith("sk-token", "get_weather", { output_policy: "trusted" });
  });

  it("should leave the row untouched and report the failure when the patch is rejected", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockRejectedValue(new Error("nope"));
    renderPanel();
    await waitForRows();

    await chooseOption(user, policySelect("tool-1", "input"), "trusted");

    await waitFor(() => expect(fromBackend).toHaveBeenCalledWith("Failed to update input policy: nope"));
    expect(policySelect("tool-1", "input").closest(".ant-select")).toHaveTextContent("untrusted");
  });

  it("should disable only the one cell that is saving", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockReturnValue(new Promise(() => {}));
    renderPanel();
    await waitForRows();

    await chooseOption(user, policySelect("tool-1", "input"), "trusted");

    await waitFor(() =>
      expect(policySelect("tool-1", "input").closest(".ant-select")).toHaveClass("ant-select-disabled"),
    );
    expect(policySelect("tool-1", "output").closest(".ant-select")).not.toHaveClass("ant-select-disabled");
    expect(policySelect("tool-2", "input").closest(".ant-select")).not.toHaveClass("ant-select-disabled");
  });
});

describe("ToolPoliciesPanel header chrome", () => {
  it("should summarise the loaded tools in the metric cards", async () => {
    renderPanel();
    await waitForRows();

    const metric = (label: string): HTMLElement => {
      const card = screen.getByText(label).closest("div.h-full");
      if (card === null) throw new Error(`metric ${label} missing`);
      return card as HTMLElement;
    };

    expect(metric("Total Tools Discovered")).toHaveTextContent("3");
    expect(metric("Blocked Tools")).toHaveTextContent("1");
    expect(metric("Active Teams")).toHaveTextContent("2");
    expect(metric("New Today")).toHaveTextContent("1");
  });

  it("should list only today's untrusted tools for review and scroll to the row", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPanel();
    await waitForRows();

    const banner = screen.getByText("Needs Review").closest("div");
    if (banner === null) throw new Error("needs review banner missing");
    expect(banner).toHaveTextContent("1 new tool discovered");
    expect(within(banner as HTMLElement).queryByText("delete_file")).not.toBeInTheDocument();

    await user.click(within(banner as HTMLElement).getByRole("button", { name: "Review" }));

    expect(row("tool-1").scrollIntoView).toHaveBeenCalled();
  });

  it("should hide the review banner when nothing needs a decision", async () => {
    fetchToolsList.mockResolvedValue([{ ...TOOLS[0], input_policy: "trusted" }]);
    renderPanel();
    await waitForRows();

    expect(screen.queryByText("Needs Review")).not.toBeInTheDocument();
  });
});

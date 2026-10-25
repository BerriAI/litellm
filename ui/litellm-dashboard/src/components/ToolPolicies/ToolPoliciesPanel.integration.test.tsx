import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { focusManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import type { ToolRow } from "@/components/networking";
import { ToolPoliciesPanel } from "./ToolPoliciesPanel";
import { toast } from "@/lib/toast";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";

const fetchToolsList = vi.fn();
const updateToolPolicy = vi.fn();

vi.mock("@/components/networking", () => ({
  fetchToolsList: (...args: unknown[]) => fetchToolsList(...args),
  updateToolPolicy: (...args: unknown[]) => updateToolPolicy(...args),
}));

const fromBackend = vi.mocked(toast.fromError);

const can = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: (...args: unknown[]) => can(...args),
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

/**
 * Exact selected-value text, read off the policy cell and stripped of anything
 * that is not a letter (the control draws a status dot and a chevron around the
 * label). Never assert with toHaveTextContent here: it substring-matches, so
 * "untrusted" satisfies "trusted".
 */
const policyValue = (toolId: string, kind: "input" | "output"): string => {
  const cell = policySelect(toolId, kind).closest("td");
  return (cell?.textContent ?? "").replace(/[^a-z]/gi, "");
};

const isSaving = (toolId: string, kind: "input" | "output"): boolean =>
  policySelect(toolId, kind).hasAttribute("disabled");

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
  can.mockReset().mockReturnValue(true);
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("ToolPoliciesPanel data loading", () => {
  it("should not fetch tools when the caller lacks the viewToolPolicies capability", async () => {
    can.mockReturnValue(false);
    renderPanel();

    await act(async () => {
      vi.advanceTimersByTime(1_000);
    });

    expect(can).toHaveBeenCalledWith("viewToolPolicies");
    expect(fetchToolsList).not.toHaveBeenCalled();
  });

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

    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");

    expect(updateToolPolicy).toHaveBeenCalledWith("sk-token", "get_weather", { input_policy: "trusted" });
    await waitFor(() => expect(policyValue("tool-1", "input")).toBe("trusted"));
    expect(fetchToolsList).toHaveBeenCalledTimes(1);
  });

  it("should patch the output policy from the output column", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPanel();
    await waitForRows();

    await chooseSelectOption(user, policySelect("tool-1", "output"), "trusted");

    expect(updateToolPolicy).toHaveBeenCalledWith("sk-token", "get_weather", { output_policy: "trusted" });
  });

  it("should keep every in-flight row disabled when two rows are saved at once", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockReturnValue(new Promise(() => {}));
    renderPanel();
    await waitForRows();

    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");
    await chooseSelectOption(user, policySelect("tool-2", "input"), "blocked");

    expect(isSaving("tool-2", "input")).toBe(true);
    expect(isSaving("tool-1", "input")).toBe(true);
  });

  it("should re-enable only the row whose save finished", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    let finishFirst = () => {};
    updateToolPolicy
      .mockImplementationOnce(() => new Promise<void>((resolve) => (finishFirst = () => resolve())))
      .mockImplementationOnce(() => new Promise(() => {}));
    renderPanel();
    await waitForRows();

    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");
    await chooseSelectOption(user, policySelect("tool-2", "input"), "blocked");
    await act(async () => {
      finishFirst();
    });

    expect(isSaving("tool-1", "input")).toBe(false);
    expect(isSaving("tool-2", "input")).toBe(true);
  });

  it("should not let an in-flight refresh clobber a policy that just saved", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    let landStaleRefresh = () => {};
    renderPanel();
    await waitForRows();

    fetchToolsList.mockImplementationOnce(
      // resolves with the PRE-save snapshot, i.e. tool-1 still "untrusted"
      () => new Promise<ToolRow[]>((resolve) => (landStaleRefresh = () => resolve(TOOLS))),
    );
    await user.click(screen.getByTestId("datatable-refresh"));
    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");
    await waitFor(() => expect(policyValue("tool-1", "input")).toBe("trusted"));

    await act(async () => {
      landStaleRefresh();
    });
    await act(async () => {
      vi.advanceTimersByTime(100);
    });

    expect(policyValue("tool-1", "input")).toBe("trusted");
  });

  it("should leave the row untouched and report the failure when the patch is rejected", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockRejectedValue(new Error("nope"));
    renderPanel();
    await waitForRows();

    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");

    await waitFor(() =>
      expect(fromBackend).toHaveBeenCalledWith(expect.objectContaining({ message: "nope" }), {
        description: expect.anything(),
      }),
    );
    expect(policyValue("tool-1", "input")).toBe("untrusted");
  });

  it("should disable only the one cell that is saving", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockReturnValue(new Promise(() => {}));
    renderPanel();
    await waitForRows();

    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");

    await waitFor(() => expect(isSaving("tool-1", "input")).toBe(true));
    expect(isSaving("tool-1", "output")).toBe(false);
    expect(isSaving("tool-2", "input")).toBe(false);
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

describe("工具策略通知的语言边界", () => {
  beforeEach(async () => {
    localStorage.clear();
    // 复用真实通知树，避免仅断言通知替身的参数而漏掉显示时的旧语言。
    const actual = await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast");
    fromBackend.mockImplementation(actual.toast.fromError);
  });

  afterEach(async () => {
    const actual = await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast");
    act(() => actual.toast.dismiss());
    fromBackend.mockReset();
  });

  const renderLocalizedPanel = (defaultLanguage = "en") =>
    renderWithProviders(
      <LanguageProvider defaultLanguage={defaultLanguage} userId="tool-policy-review" storageScope="/tool-policies">
        <LanguageSelector />
        <ToolPoliciesPanel accessToken="sk-token" onSelectTool={vi.fn()} />
        <Toaster />
      </LanguageProvider>,
    );

  it.each([
    ["input", "更新输入策略失败：Home", "Failed to update input policy: Home"],
    ["output", "更新输出策略失败：Home", "Failed to update output policy: Home"],
  ] as const)("%s 策略请求等待中及失败提示显示后均跟随当前语言，业务原文不翻译", async (kind, chinese, english) => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const pending = Promise.withResolvers<void>();
    updateToolPolicy.mockReturnValue(pending.promise);
    renderLocalizedPanel();
    await waitForRows();

    await chooseSelectOption(user, policySelect("tool-1", kind), "trusted");
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(isSaving("tool-1", kind)).toBe(true);
    await act(async () => pending.reject(new Error("Home")));

    expect(await screen.findByText(chinese)).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByText(english)).toBeInTheDocument();
    expect(policyValue("tool-1", kind)).toBe("untrusted");
    expect(updateToolPolicy).toHaveBeenCalledExactlyOnceWith("sk-token", "get_weather", {
      [`${kind}_policy`]: "trusted",
    });
    expect(fetchToolsList).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["input", "更新输入策略失败：未知错误", "Failed to update input policy: unknown error"],
    ["output", "更新输出策略失败：未知错误", "Failed to update output policy: unknown error"],
  ] as const)("%s 策略本地兜底不夹带英文，提示显示后仍可切换", async (kind, chinese, english) => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockRejectedValue(null);
    renderLocalizedPanel();
    await waitForRows();
    await chooseSelectOption(user, policySelect("tool-1", kind), "trusted");

    expect(await screen.findByText(english)).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByText(chinese)).toBeInTheDocument();
    expect(updateToolPolicy).toHaveBeenCalledTimes(1);
    expect(fetchToolsList).toHaveBeenCalledTimes(1);
  });

  it("只翻译列表的本地加载兜底，不因切换语言重新查询", async () => {
    fetchToolsList.mockRejectedValue("failed");
    renderLocalizedPanel();

    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to load tools");
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByRole("alert")).toHaveTextContent("工具加载失败");
    expect(fetchToolsList).toHaveBeenCalledTimes(1);
  });

  it.each(["Home", "unknown error", "Failed to load tools"])("列表服务端原文 %s 不作为翻译键", async (message) => {
    fetchToolsList.mockRejectedValue(new Error(message));
    renderLocalizedPanel("zh-CN");

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByRole("alert")).toHaveTextContent(message);
    expect(fetchToolsList).toHaveBeenCalledTimes(1);
  });

  it("与本地兜底同名的策略服务端详情保留原文", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockRejectedValue(new Error("unknown error"));
    renderLocalizedPanel();
    await waitForRows();
    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");
    expect(await screen.findByText("Failed to update input policy: unknown error")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByText("更新输入策略失败：unknown error")).toBeInTheDocument();
  });

  it("已识别的服务端错误包装保留权限分类和业务详情，不暴露调试字段", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    updateToolPolicy.mockRejectedValue(
      new Error(JSON.stringify({ error: { message: "Home", code: 403 }, trace: "private-debug-detail" })),
    );
    renderLocalizedPanel();
    await waitForRows();
    await chooseSelectOption(user, policySelect("tool-1", "input"), "trusted");

    expect(await screen.findByText("Access Denied")).toBeInTheDocument();
    expect(screen.getByText("Failed to update input policy: Home")).toBeInTheDocument();
    expect(screen.queryByText(/private-debug-detail/)).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByText("无访问权限")).toBeInTheDocument();
    expect(screen.getByText("更新输入策略失败：Home")).toBeInTheDocument();
  });
});

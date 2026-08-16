import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getClaudeCodePluginsList } from "@/components/networking";

import ClaudeCodePluginsPanel from "./ClaudeCodePluginsPanel";

vi.mock("@/components/networking", () => ({
  getClaudeCodePluginsList: vi.fn(),
  deleteClaudeCodePlugin: vi.fn(),
}));

vi.mock("./PluginTable", () => ({
  __esModule: true,
  default: ({ isLoading }: { isLoading: boolean }) => (
    <div data-testid="plugin-table">{isLoading ? "table-loading" : "table-loaded"}</div>
  ),
}));

vi.mock("./add_plugin_form", () => ({ __esModule: true, default: () => null }));
vi.mock("@/components/claude_code_plugins/skill_detail", () => ({ __esModule: true, default: () => null }));

const mockGetClaudeCodePluginsList = vi.mocked(getClaudeCodePluginsList);

describe("ClaudeCodePluginsPanel loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    render(<ClaudeCodePluginsPanel accessToken={null} />);
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetClaudeCodePluginsList).not.toHaveBeenCalled();
  });

  it("should show the loading state until the skills fetch settles", async () => {
    let resolveFetch: (value: { plugins: never[]; count: number }) => void = () => {};
    mockGetClaudeCodePluginsList.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ plugins: [], count: 0 });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetClaudeCodePluginsList).toHaveBeenCalledWith("sk-test", false);
  });
});

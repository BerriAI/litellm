import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ToolDetail } from "./ToolDetail";
import {
  deleteToolPolicyOverride,
  fetchToolDetail,
  fetchToolPolicyOptions,
  getToolUsageLogs,
  keyListCall,
  teamListCall,
  updateToolPolicy,
  type ToolDetailResponse,
  type ToolPolicyOption,
  type ToolPolicyOverrideRow,
  type ToolRow,
  type ToolUsageLogsResponse,
} from "@/components/networking";

vi.mock("@/components/networking", () => ({
  deleteToolPolicyOverride: vi.fn(),
  fetchToolDetail: vi.fn(),
  fetchToolPolicyOptions: vi.fn(),
  getToolUsageLogs: vi.fn(),
  keyListCall: vi.fn(),
  teamListCall: vi.fn(),
  updateToolPolicy: vi.fn(),
}));

vi.mock("@/components/common_components/team_dropdown", () => ({
  default: ({ onChange }: { onChange: (id: string) => void }) => (
    <button type="button" onClick={() => onChange("team-1")}>
      pick team
    </button>
  ),
}));

vi.mock("@/components/GuardrailsMonitor/LogViewer", () => ({
  LogViewer: ({ totalLogs }: { totalLogs: number }) => <div>log viewer ({totalLogs})</div>,
}));

const detail = {
  tool: {
    tool_name: "search_docs",
    input_policy: "untrusted",
    output_policy: "trusted",
    origin: "mcp",
    call_count: 42,
    user_agent: "litellm-python/1.0",
    created_at: "2026-03-04T10:00:00Z",
  },
  overrides: [],
} as unknown as ToolDetailResponse;

const renderDetail = (onBack = vi.fn()) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ToolDetail toolName="search_docs" onBack={onBack} accessToken="tok" />
    </QueryClientProvider>,
  );
};

describe("ToolDetail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchToolDetail).mockResolvedValue(detail);
    vi.mocked(fetchToolPolicyOptions).mockResolvedValue({ input_policies: [], output_policies: [] });
    vi.mocked(teamListCall).mockResolvedValue({ data: [] });
    vi.mocked(keyListCall).mockResolvedValue({ keys: [] });
    vi.mocked(getToolUsageLogs).mockResolvedValue({ logs: [], total: 0 } as unknown as ToolUsageLogsResponse);
    vi.mocked(updateToolPolicy).mockResolvedValue(undefined as unknown as ToolRow);
    vi.mocked(deleteToolPolicyOverride).mockResolvedValue(
      undefined as unknown as { deleted: boolean; tool_name: string },
    );
  });

  it("shows the tool identity once loaded", async () => {
    renderDetail();

    expect(await screen.findByText("search_docs")).toBeInTheDocument();
    expect(screen.getByText("mcp")).toBeInTheDocument();
    expect(screen.getByText("42 calls")).toBeInTheDocument();
    expect(screen.getByText("litellm-python/1.0")).toBeInTheDocument();
  });

  it("renders both policy panels with the tool's current policies", async () => {
    renderDetail();

    expect(await screen.findByText("Input Policy")).toBeInTheDocument();
    expect(screen.getByText("Output Policy")).toBeInTheDocument();
    expect(screen.getByText("untrusted")).toBeInTheDocument();
    expect(screen.getByText("trusted")).toBeInTheDocument();
  });

  it("uses the policy option descriptions when the backend supplies them", async () => {
    vi.mocked(fetchToolPolicyOptions).mockResolvedValue({
      input_policies: [{ value: "untrusted", description: "Treat inputs as hostile" } as ToolPolicyOption],
      output_policies: [{ value: "trusted", description: "Outputs may be chained" } as ToolPolicyOption],
    });

    renderDetail();

    expect(await screen.findByText("Treat inputs as hostile")).toBeInTheDocument();
    expect(screen.getByText("Outputs may be chained")).toBeInTheDocument();
  });

  it("returns to the list when Back is pressed", async () => {
    const onBack = vi.fn();
    renderDetail(onBack);

    await userEvent.click(await screen.findByRole("button", { name: /Back to Tool Policies/ }));

    expect(onBack).toHaveBeenCalled();
  });

  it("reports a failed detail load and still offers a way back", async () => {
    vi.mocked(fetchToolDetail).mockRejectedValue(new Error("nope"));

    renderDetail();

    expect(await screen.findByText("Failed to load tool details.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Back to Tool Policies/ })).toBeInTheDocument();
  });

  it("hides the overrides panel when the tool has none", async () => {
    renderDetail();

    await screen.findByText("Input Policy");
    expect(screen.queryByText("Blocked for team or key")).not.toBeInTheDocument();
  });

  it("lists existing overrides and removes the chosen one", async () => {
    vi.mocked(fetchToolDetail).mockResolvedValue({
      ...detail,
      overrides: [
        {
          override_id: "o1",
          team_id: "team-alpha",
          key_hash: null,
          key_alias: null,
        } as unknown as ToolPolicyOverrideRow,
      ],
    });

    renderDetail();

    expect(await screen.findByText("Team: team-alpha")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Remove" }));

    await waitFor(() =>
      expect(deleteToolPolicyOverride).toHaveBeenCalledWith("tok", "search_docs", {
        team_id: "team-alpha",
        key_hash: undefined,
      }),
    );
  });

  it("keeps the block button disabled until a team is chosen, then blocks that team", async () => {
    renderDetail();

    const blockButton = await screen.findByRole("button", { name: /Block for team/ });
    expect(blockButton).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: "pick team" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /Block for team/ })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: /Block for team/ }));

    await waitFor(() =>
      expect(updateToolPolicy).toHaveBeenCalledWith(
        "tok",
        "search_docs",
        { input_policy: "blocked" },
        { team_id: "team-1", key_hash: undefined, key_alias: undefined },
      ),
    );
  });

  it("switches the block scope to a key", async () => {
    renderDetail();

    await screen.findByText("Block for team or key");
    await userEvent.click(screen.getByRole("radio", { name: "Key" }));

    expect(await screen.findByRole("button", { name: /Block for key/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "pick team" })).not.toBeInTheDocument();
  });

  it("passes the usage-log total through to the log viewer", async () => {
    vi.mocked(getToolUsageLogs).mockResolvedValue({ logs: [], total: 7 } as unknown as ToolUsageLogsResponse);

    renderDetail();

    expect(await screen.findByText("log viewer (7)")).toBeInTheDocument();
  });
});

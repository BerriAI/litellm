import { useState, type ComponentProps } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { uiSpendLogsCall, userListCall, dailyActivityExportCall, type UserInfo } from "@/components/networking";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import {
  chooseSelectOption,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../../tests/test-utils";
import ClientUsagePanel from "./ClientUsagePanel";
import { UsageClient, type AgentRow } from "./overview/agentCatalog";

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  uiSpendLogsCall: vi.fn(),
  dailyActivityExportCall: vi.fn(),
  userListCall: vi.fn(),
}));

const desktopTag = "User-Agent: claude-cli/2.1.263 (external, claude-desktop-3p)";
const agent: AgentRow = {
  id: "claude-desktop",
  label: "Claude Desktop",
  description: "Desktop assistant",
  kind: "agent",
  logo: null,
  spend: 12.5,
  tokens: 9000,
  requests: 3,
  users: 2,
  tags: [desktopTag],
};
const metrics: SpendMetrics = {
  spend: 12.5,
  total_tokens: 9000,
  prompt_tokens: 8000,
  completion_tokens: 1000,
  cache_read_input_tokens: 5000,
  cache_creation_input_tokens: 1000,
  api_requests: 3,
  successful_requests: 3,
  failed_requests: 0,
};
const daily: DailyData[] = [
  {
    date: "2026-10-22",
    metrics,
    breakdown: {
      models: {},
      model_groups: {},
      providers: {},
      api_keys: {},
      mcp_servers: {},
      entities: { [desktopTag]: { metrics, metadata: {}, api_key_breakdown: {} } },
    },
  },
];
const defaultProps: ComponentProps<typeof ClientUsagePanel> = {
  client: UsageClient.ClaudeDesktop,
  userId: null,
  canChooseUser: true,
  onUserChange: vi.fn(),
  onClose: vi.fn(),
  accessToken: "test-token",
  startTime: new Date("2026-10-22T00:00:00Z"),
  endTime: new Date("2026-10-22T23:59:59.999Z"),
  dateValue: { from: new Date("2026-10-22T00:00:00Z"), to: new Date("2026-10-22T23:59:59.999Z") },
  presetShortLabel: null,
  onDateChange: vi.fn(),
  agent,
  daily: [],
  loading: false,
};
const personLog = {
  request_id: "request-1",
  user: "person-1",
  startTime: "2026-10-22T12:00:00Z",
  model: "test-model",
  spend: 3,
  total_tokens: 1000,
  prompt_tokens: 800,
  completion_tokens: 200,
  status: "success",
  request_tags: [desktopTag],
  metadata: {},
};
const person: UserInfo = {
  user_id: "person-1",
  user_email: "person@example.com",
  user_alias: null,
  allowed_cache_controls: [],
  key_count: 0,
  model_max_budget: {},
  model_spend: {},
  models: [],
  policies: [],
  spend: 0,
  teams: [],
};

describe("ClientUsagePanel", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) => {
      callback({ didTimeout: false, timeRemaining: () => 50 });
      return 1;
    });
    const userPage = {
      users: [person],
      total: 1,
      page: 1,
      page_size: 50,
      total_pages: 1,
    };
    const requestPage = {
      data: [personLog],
      total: 1,
      page: 1,
      total_pages: 1,
      total_is_capped: false,
    };
    const exportedRows = [
      {
        date: "2026-10-22",
        entity_id: desktopTag,
        model: "test-model",
        spend: 12.5,
        flat_cost: 0,
        prompt_tokens: 8000,
        completion_tokens: 1000,
        api_requests: 3,
        successful_requests: 3,
        failed_requests: 0,
        cache_read_input_tokens: 5000,
        cache_creation_input_tokens: 1000,
      },
    ];
    vi.mocked(dailyActivityExportCall).mockResolvedValue(
      Object.assign(new Blob([JSON.stringify(exportedRows)], { type: "application/json" }), {
        text: async () => JSON.stringify(exportedRows),
      }),
    );
    vi.mocked(userListCall).mockResolvedValue(userPage);
    vi.mocked(uiSpendLogsCall).mockResolvedValue(requestPage);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({ results: [] })),
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  it("should immediately show the clicked all-users summary and logo without requiring a person", () => {
    renderWithProviders(<ClientUsagePanel {...defaultProps} loading />);

    const panel = screen.getByRole("dialog", { name: "Claude Desktop" });
    expect(within(panel).getByRole("img", { name: "Claude Desktop logo" })).toBeInTheDocument();
    expect(panel).toHaveTextContent("All users");
    expect(panel).toHaveTextContent(/Spend\s*\$12\.50/);
    expect(panel).toHaveTextContent(/Total tokens\s*9K/);
    expect(panel).toHaveTextContent(/Requests\s*3/);
    expect(screen.queryByRole("status", { name: "Loading client usage" })).not.toBeInTheDocument();
    expect(screen.queryByText(/Select a user to/)).not.toBeInTheDocument();
    expect(uiSpendLogsCall).not.toHaveBeenCalled();
  });

  it("should filter to a person inside the open panel and keep absent cache details unavailable", async () => {
    const user = userEvent.setup();
    const onUserChange = vi.fn();
    const onClose = vi.fn();
    function ControlledPanel() {
      const [userId, setUserId] = useState<string | null>(null);
      return (
        <ClientUsagePanel
          {...defaultProps}
          canChooseUser
          userId={userId}
          onClose={onClose}
          onUserChange={(id) => {
            setUserId(id);
            onUserChange(id);
          }}
        />
      );
    }
    renderWithProviders(<ControlledPanel />);

    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter by person" }), /person@example.com/);
    expect(onUserChange).toHaveBeenCalledWith("person-1");
    expect(onClose).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(uiSpendLogsCall).toHaveBeenCalledWith(
        expect.objectContaining({
          params: expect.objectContaining({ user_id: "person-1" }),
        }),
      ),
    );
    const panel = screen.getByRole("dialog", { name: "Claude Desktop" });
    await waitFor(() => expect(panel).toHaveTextContent(/Spend\s*\$3\.00/));
    expect(panel).toHaveTextContent("Selected user");
    expect(panel).toHaveTextContent(/Cache reads\s*Unavailable/);
    expect(panel).toHaveTextContent(/Cache writes\s*Unavailable/);
    expect(panel).toHaveTextContent(/Uncached input\s*Unavailable/);
    expect(panel).toHaveTextContent(/Output\s*200/);

    await user.click(screen.getByRole("button", { name: "All users" }));
    expect(onUserChange).toHaveBeenLastCalledWith(null);
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Claude Desktop" })).toHaveTextContent(/Spend\s*\$12\.50/);
  });

  it.each(["Back to Usage", "Close"])("should return to the overview through %s", async (button) => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithProviders(<ClientUsagePanel {...defaultProps} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: button }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("should preserve summary totals and mark partial model details unavailable", async () => {
    const partialAgent = { ...agent, spend: 25, tokens: 18000, requests: 6 };
    renderWithProviders(<ClientUsagePanel {...defaultProps} agent={partialAgent} daily={daily} />);

    const panel = screen.getByRole("dialog", { name: "Claude Desktop" });
    expect(panel).toHaveTextContent(/Spend\s*\$25\.00/);
    expect(within(panel).getAllByText("Unavailable")).toHaveLength(4);
    expect(await screen.findByText(/Model usage does not match these client totals yet/)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Daily usage" })).not.toBeInTheDocument();
    expect(screen.queryByText("5,000")).not.toBeInTheDocument();
  });

  it("should not expose a cached all-users summary without a permitted user scope", () => {
    renderWithProviders(<ClientUsagePanel {...defaultProps} canChooseUser={false} userId={null} daily={daily} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Sign in to view your client usage");
    expect(screen.queryByText("$12.50")).not.toBeInTheDocument();
    expect(screen.queryByText("Total tokens")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Filter by person" })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Daily usage" })).not.toBeInTheDocument();
    expect(uiSpendLogsCall).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });

  it("should show a failed summary as an error and retry without closing or showing stale totals", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const onClose = vi.fn();
    renderWithProviders(<ClientUsagePanel {...defaultProps} summaryFailed onRetry={onRetry} onClose={onClose} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Could not load client totals");
    expect(screen.queryByText("$12.50")).not.toBeInTheDocument();
    expect(screen.queryByText("No Claude Desktop usage recorded in this date range.")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledOnce();
    expect(onClose).not.toHaveBeenCalled();
  });
  it("should change the panel time range through the existing picker without closing", async () => {
    const user = userEvent.setup();
    const onDateChange = vi.fn();
    const onClose = vi.fn();
    renderWithProviders(<ClientUsagePanel {...defaultProps} onDateChange={onDateChange} onClose={onClose} />);
    await user.click(screen.getByRole("button", { name: /22 Oct/ }));
    await user.click(screen.getByRole("button", { name: /Last 7 days/ }));
    await user.click(screen.getByRole("button", { name: "Apply" }));
    expect(onDateChange).toHaveBeenCalledWith(
      expect.objectContaining({ from: expect.any(Date), to: expect.any(Date) }),
      "7d",
    );
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Claude Desktop" })).toBeInTheDocument();
  });

  it("should show the client model breakdown using the overview controls", async () => {
    renderWithProviders(<ClientUsagePanel {...defaultProps} />);
    expect(await screen.findByText("test-model")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Daily usage" })).toBeInTheDocument();
    expect(screen.getByRole("radiogroup", { name: "Metric" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Tokens" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Weekly" })).toBeInTheDocument();
  });

  it("should show personal client usage without loading the global user list or all-user export", async () => {
    renderWithProviders(<ClientUsagePanel {...defaultProps} userId="person-1" canChooseUser={false} />);

    const panel = screen.getByRole("dialog", { name: "Claude Desktop" });
    await waitFor(() => expect(panel).toHaveTextContent(/Spend\s*\$3\.00/));
    expect(panel).toHaveTextContent("My usage");
    expect(screen.queryByRole("combobox", { name: "Filter by person" })).not.toBeInTheDocument();
    expect(userListCall).not.toHaveBeenCalled();
    expect(dailyActivityExportCall).not.toHaveBeenCalled();
    expect(uiSpendLogsCall).toHaveBeenCalledWith(
      expect.objectContaining({ params: expect.objectContaining({ user_id: "person-1" }) }),
    );
    expect(vi.mocked(uiSpendLogsCall).mock.lastCall?.[0].params).not.toHaveProperty("team_id");
  });

  it("should keep team and member filters intersected inside the open panel without fetching global users", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onTeamChange = vi.fn();
    const onUserChange = vi.fn();
    const teamOptions = [{ value: "engineering", label: "Engineering" }];
    const memberOptions = [
      { value: "person-1", label: "John" },
      { value: "person-2", label: "Maya" },
    ];
    const johnLog = { ...personLog, team_id: "engineering" };
    const mayaLog = { ...johnLog, request_id: "request-2", user: "person-2", spend: 7 };
    vi.mocked(uiSpendLogsCall).mockImplementation(async (request) => {
      const logs = [johnLog, mayaLog].filter((log) => !request.params?.user_id || log.user === request.params.user_id);
      return { data: logs, total: logs.length, page: 1, total_pages: 1, total_is_capped: false };
    });
    function TeamPanel() {
      const [teamId, setTeamId] = useState<string | null>(null);
      const [userId, setUserId] = useState<string | null>("person-1");
      const changeTeam = (id: string | null) => {
        setTeamId(id);
        setUserId(id ? null : "person-1");
        onTeamChange(id);
      };
      return (
        <ClientUsagePanel
          {...defaultProps}
          canChooseUser={false}
          defaultScopeLabel="My usage"
          teams={teamOptions}
          members={teamId ? memberOptions : []}
          teamId={teamId}
          userId={userId}
          onClose={onClose}
          onTeamChange={changeTeam}
          onUserChange={(id) => {
            setUserId(id);
            onUserChange(id);
          }}
        />
      );
    }
    renderWithProviders(<TeamPanel />);

    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Usage scope" }), "Engineering");
    expect(onTeamChange).toHaveBeenCalledWith("engineering");
    const panel = screen.getByRole("dialog", { name: "Claude Desktop" });
    await waitFor(() => expect(panel).toHaveTextContent(/Spend\s*\$10\.00/));
    expect(panel).toHaveTextContent("Engineering · All members");
    expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
      expect.objectContaining({ params: expect.objectContaining({ team_id: "engineering" }) }),
    );
    expect(vi.mocked(uiSpendLogsCall).mock.lastCall?.[0].params).not.toHaveProperty("user_id");

    await user.click(screen.getByRole("combobox", { name: "Team member" }));
    expect(await screen.findByRole("option", { name: "Maya" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "John" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Outside user" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: "Maya" }));
    expect(onUserChange).toHaveBeenCalledWith("person-2");
    await waitFor(() => expect(panel).toHaveTextContent(/Spend\s*\$7\.00/));
    expect(panel).toHaveTextContent("Engineering · Maya");
    expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
      expect.objectContaining({ params: expect.objectContaining({ team_id: "engineering", user_id: "person-2" }) }),
    );

    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Team member" }), "All members");
    expect(onUserChange).toHaveBeenLastCalledWith(null);
    await waitFor(() => expect(panel).toHaveTextContent(/Spend\s*\$10\.00/));
    expect(onClose).not.toHaveBeenCalled();
    expect(userListCall).not.toHaveBeenCalled();
    expect(dailyActivityExportCall).not.toHaveBeenCalled();
  });

  it.each([
    [null, "Choose an administered team before viewing another member's usage"],
    ["engineering", "Choose a member of the selected team"],
  ])("should block requests and cached summaries for an invalid scope in team %s", (teamId, scopeError) => {
    renderWithProviders(
      <ClientUsagePanel
        {...defaultProps}
        canChooseUser={false}
        teamId={teamId}
        userId="outside-user"
        scopeError={scopeError}
        daily={daily}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent(scopeError);
    expect(screen.queryByText("$12.50")).not.toBeInTheDocument();
    expect(screen.queryByText("Total tokens")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Daily usage" })).not.toBeInTheDocument();
    expect(uiSpendLogsCall).not.toHaveBeenCalled();
    expect(dailyActivityExportCall).not.toHaveBeenCalled();
    expect(userListCall).not.toHaveBeenCalled();
  });
  it("should recover a newer model export by refreshing the cached summary on retry", async () => {
    const user = userEvent.setup();
    function StaleSummaryPanel() {
      const [summary, setSummary] = useState({ ...agent, spend: 12 });
      return <ClientUsagePanel {...defaultProps} agent={summary} onRetry={() => setSummary(agent)} />;
    }
    renderWithProviders(<StaleSummaryPanel />);
    expect(await screen.findByText(/Model usage does not match these client totals/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry model breakdown" }));
    expect(await screen.findByText("test-model")).toBeInTheDocument();
  });
});

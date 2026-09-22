import React from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MCPGatewaySessionsTab, describeTerminateResult, formatIdleSeconds } from "./MCPGatewaySessionsTab";
import * as networking from "@/components/networking";
import type { MCPGatewaySessionsResponse, MCPGatewaySessionsTerminateResponse } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  fetchMCPGatewaySessions: vi.fn(),
  terminateMCPGatewaySessions: vi.fn(),
}));

const REPORT: MCPGatewaySessionsResponse = {
  worker_pid: 4242,
  total_sessions: 3,
  by_client: [
    { label: "claude-code", count: 2 },
    { label: "cursor", count: 1 },
  ],
  by_user: [
    { label: "alice", count: 2 },
    { label: null, count: 1 },
  ],
  sessions: [
    {
      session_id_prefix: "aaaa1111",
      client_name: "claude-code",
      client_version: "1.0.0",
      user_id: "alice",
      user_email: "alice@example.com",
      key_alias: "alice-key",
      team_id: "team-1",
      team_alias: "platform",
      client_ip: "10.0.0.1",
      idle_seconds: 75,
      in_flight_requests: 0,
    },
    {
      session_id_prefix: "bbbb2222",
      client_name: "claude-code",
      client_version: "1.0.1",
      user_id: "alice",
      user_email: null,
      key_alias: null,
      team_id: null,
      team_alias: null,
      client_ip: "",
      idle_seconds: 3,
      in_flight_requests: 1,
    },
    {
      session_id_prefix: "cccc3333",
      client_name: "cursor",
      client_version: null,
      user_id: null,
      user_email: null,
      key_alias: null,
      team_id: null,
      team_alias: null,
      client_ip: null,
      idle_seconds: 0,
      in_flight_requests: 0,
    },
  ],
};

const renderTab = ({ canTerminate = false }: { canTerminate?: boolean } = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MCPGatewaySessionsTab accessToken="token" canTerminate={canTerminate} />
    </QueryClientProvider>,
  );
};

describe("formatIdleSeconds", () => {
  it("renders seconds under a minute and minutes plus seconds above it", () => {
    expect(formatIdleSeconds(0)).toBe("0s");
    expect(formatIdleSeconds(59.9)).toBe("59s");
    expect(formatIdleSeconds(60)).toBe("1m");
    expect(formatIdleSeconds(75)).toBe("1m 15s");
    expect(formatIdleSeconds(-4)).toBe("0s");
  });
});

describe("describeTerminateResult", () => {
  it("pluralizes the session count and names the worker", () => {
    expect(describeTerminateResult({ worker_pid: 9, terminated_sessions: 1, sessions: [] })).toBe(
      "Disconnected 1 session on worker pid 9.",
    );
    expect(describeTerminateResult({ worker_pid: 9, terminated_sessions: 0, sessions: [] })).toBe(
      "Disconnected 0 sessions on worker pid 9.",
    );
  });
});

describe("MCPGatewaySessionsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows grouped counts and session rows from /v1/mcp/sessions", async () => {
    vi.mocked(networking.fetchMCPGatewaySessions).mockResolvedValue(REPORT);
    renderTab();

    const byClient = await screen.findByRole("region", { name: "Sessions by AI client" });
    expect(within(byClient).getByRole("row", { name: /claude-code 2/ })).toBeInTheDocument();
    expect(within(byClient).getByRole("row", { name: /cursor 1/ })).toBeInTheDocument();

    const byUser = screen.getByRole("region", { name: "Sessions by user" });
    expect(within(byUser).getByRole("row", { name: /alice 2/ })).toBeInTheDocument();
    expect(within(byUser).getByRole("row", { name: /\(unknown\) 1/ })).toBeInTheDocument();

    const sessions = screen.getByRole("region", { name: "Live sessions" });
    const firstRow = within(sessions).getByRole("row", { name: /aaaa1111/ });
    expect(firstRow).toHaveTextContent("claude-code");
    expect(firstRow).toHaveTextContent("v1.0.0");
    expect(firstRow).toHaveTextContent("alice@example.com");
    expect(firstRow).toHaveTextContent("platform");
    expect(firstRow).toHaveTextContent("1m 15s");
    expect(within(sessions).getByRole("row", { name: /cccc3333/ })).toHaveTextContent("(unknown)");
    expect(screen.getByText("Live sessions (worker pid 4242)")).toBeInTheDocument();
    expect(networking.fetchMCPGatewaySessions).toHaveBeenCalledWith("token");
  });

  it("shows an empty state when the worker holds no live sessions", async () => {
    const emptyReport: MCPGatewaySessionsResponse = {
      worker_pid: 7,
      total_sessions: 0,
      by_client: [],
      by_user: [],
      sessions: [],
    };
    vi.mocked(networking.fetchMCPGatewaySessions).mockResolvedValue(emptyReport);
    renderTab();

    expect(await screen.findByText(/No live MCP connections on this worker \(pid 7\)/)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Live sessions" })).not.toBeInTheDocument();
  });

  it("shows the API error when the request fails", async () => {
    vi.mocked(networking.fetchMCPGatewaySessions).mockRejectedValue(new Error("Admin access required"));
    renderTab();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not load live connections");
    expect(alert).toHaveTextContent("Admin access required");
  });

  it("hides every disconnect control from a read-only admin", async () => {
    vi.mocked(networking.fetchMCPGatewaySessions).mockResolvedValue(REPORT);
    renderTab({ canTerminate: false });

    await screen.findByRole("region", { name: "Live sessions" });
    expect(screen.queryByRole("button", { name: /^Disconnect/ })).not.toBeInTheDocument();
  });

  it("disconnects one session by its displayed prefix after confirmation and refetches", async () => {
    const user = userEvent.setup();
    const terminated: MCPGatewaySessionsTerminateResponse = {
      worker_pid: 4242,
      terminated_sessions: 1,
      sessions: [REPORT.sessions[1]],
    };
    vi.mocked(networking.fetchMCPGatewaySessions).mockResolvedValue(REPORT);
    vi.mocked(networking.terminateMCPGatewaySessions).mockResolvedValue(terminated);
    renderTab({ canTerminate: true });

    await user.click(await screen.findByRole("button", { name: "Disconnect session bbbb2222" }));
    expect(networking.terminateMCPGatewaySessions).not.toHaveBeenCalled();
    const dialog = await screen.findByRole("alertdialog");
    expect(dialog).toHaveTextContent("session bbbb2222");
    await user.click(within(dialog).getByRole("button", { name: "Disconnect" }));

    const status = await screen.findByText("Disconnected 1 session on worker pid 4242.", { exact: false });
    expect(status).toBeInTheDocument();
    expect(networking.terminateMCPGatewaySessions).toHaveBeenCalledWith("token", { session_id_prefix: "bbbb2222" });
    expect(networking.fetchMCPGatewaySessions).toHaveBeenCalledTimes(2);
  });

  it("disconnects every session of a user from the by-user table", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.fetchMCPGatewaySessions).mockResolvedValue(REPORT);
    vi.mocked(networking.terminateMCPGatewaySessions).mockResolvedValue({
      worker_pid: 4242,
      terminated_sessions: 2,
      sessions: [REPORT.sessions[0], REPORT.sessions[1]],
    });
    renderTab({ canTerminate: true });

    const byUser = await screen.findByRole("region", { name: "Sessions by user" });
    expect(within(byUser).queryByRole("button", { name: /\(unknown\)/ })).not.toBeInTheDocument();
    await user.click(within(byUser).getByRole("button", { name: "Disconnect all sessions for user alice" }));
    const dialog = await screen.findByRole("alertdialog");
    expect(dialog).toHaveTextContent("every live session opened by user alice");
    await user.click(within(dialog).getByRole("button", { name: "Disconnect" }));

    expect(await screen.findByText(/Disconnected 2 sessions on worker pid 4242\./)).toBeInTheDocument();
    expect(networking.terminateMCPGatewaySessions).toHaveBeenCalledWith("token", { user_id: "alice" });
  });

  it("shows the API error when a disconnect is refused", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.fetchMCPGatewaySessions).mockResolvedValue(REPORT);
    vi.mocked(networking.terminateMCPGatewaySessions).mockRejectedValue(
      new Error("Proxy admin access required to terminate MCP gateway sessions."),
    );
    renderTab({ canTerminate: true });

    await user.click(await screen.findByRole("button", { name: "Disconnect session aaaa1111" }));
    await user.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Disconnect" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not disconnect");
    expect(alert).toHaveTextContent("Proxy admin access required to terminate MCP gateway sessions.");
    expect(screen.getByRole("region", { name: "Live sessions" })).toBeInTheDocument();
  });
});

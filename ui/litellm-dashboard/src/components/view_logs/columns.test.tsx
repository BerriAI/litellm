import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { createColumns, type LogEntry } from "./columns";
import { DataTable } from "./table";

const logEntry = (overrides: Partial<LogEntry>): LogEntry => ({
  request_id: "req-1",
  api_key: "key-1",
  team_id: "team-1",
  model: "gpt-4o",
  model_id: "model-1",
  call_type: "acompletion",
  spend: 0,
  total_tokens: 10,
  prompt_tokens: 5,
  completion_tokens: 5,
  startTime: "2026-07-07T09:50:13Z",
  endTime: "2026-07-07T09:50:14Z",
  cache_hit: "false",
  messages: [],
  response: {},
  ...overrides,
});

describe("Cost column", () => {
  it("renders '-' for zero spend with no tooltip, so hovering never shows a contradictory $0", async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        data={[logEntry({ request_id: "req-zero", spend: 0 })]}
        columns={createColumns()}
        getRowId={(r) => r.request_id}
      />,
    );
    for (const dash of screen.getAllByText("-")) {
      await user.hover(dash);
    }
    expect(screen.queryByText("$0")).not.toBeInTheDocument();
  });

  it("shows the full-precision raw value in the tooltip for a real spend", async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        data={[logEntry({ request_id: "req-spend", spend: 0.00012345678 })]}
        columns={createColumns()}
        getRowId={(r) => r.request_id}
      />,
    );
    const formatted = screen.getByText("$0.000123");
    await user.hover(formatted);
    expect(await screen.findByText("$0.00012345678")).toBeInTheDocument();
  });

  it("shows the summed session total, not the representative call's spend, for a multi-round session", () => {
    const overrides: Partial<LogEntry> = {
      request_id: "req-session",
      spend: 0.01,
      session_id: "sess-1",
      session_total_count: 3,
      session_total_spend: 0.06,
    };
    render(<DataTable data={[logEntry(overrides)]} columns={createColumns()} getRowId={(r) => r.request_id} />);
    expect(screen.getByText("$0.060000")).toBeInTheDocument();
    expect(screen.queryByText("$0.010000")).not.toBeInTheDocument();
    expect(screen.getByText("session total")).toBeInTheDocument();
  });
});

describe("view logs columns", () => {
  const relayLog = logEntry({
    request_id: "req-relay",
    api_key: "hashed-relay-key",
    team_id: "",
    model: "notion-ai",
    api_base: "https://www.notion.so",
    call_type: "litellm-relay",
    total_tokens: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    metadata: {
      app: "notion",
      status: "success",
      status_code: 200,
      user_api_key: null,
      user_api_key_alias: "relay-key",
      user_api_key_team_alias: null,
    },
    request_tags: { source: "notion" },
    proxy_server_request: {},
    status: "success",
  });

  it("should render relay type with captured app logo and name", () => {
    render(<DataTable data={[relayLog]} columns={createColumns()} getRowId={(row) => row.request_id} />);

    expect(screen.getByText("litellm-relay")).toBeInTheDocument();
    expect(screen.getAllByText("Notion").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("img", { name: "Notion logo" }).length).toBeGreaterThan(0);
    expect(screen.queryByRole("columnheader", { name: "Source" })).not.toBeInTheDocument();
    expect(screen.queryByText("LLM")).not.toBeInTheDocument();
  });

  it("should render collector rows with relay metadata as relay logs", () => {
    const legacyCollectorLog = {
      ...relayLog,
      request_id: "collector-01f159da",
      call_type: "completion",
      metadata: {
        ...relayLog.metadata,
        app: "codex",
        source: "litellm-relay",
      },
      model: "codex-ai",
      request_tags: ["litellm-relay", "codex"],
    };

    render(<DataTable data={[legacyCollectorLog]} columns={createColumns()} getRowId={(row) => row.request_id} />);

    expect(screen.getByText("litellm-relay")).toBeInTheDocument();
    expect(screen.getAllByText("Codex").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("img", { name: "Codex logo" }).length).toBeGreaterThan(0);
    expect(screen.queryByText("LLM")).not.toBeInTheDocument();
  });

  it("should fall back to row api_key when relay metadata does not include a key hash", () => {
    render(<DataTable data={[relayLog]} columns={createColumns()} getRowId={(row) => row.request_id} />);

    expect(screen.getByText("hashed-relay-key")).toBeInTheDocument();
  });
});

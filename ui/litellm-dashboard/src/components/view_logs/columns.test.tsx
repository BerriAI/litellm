import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createColumns, LogEntry } from "./columns";
import { DataTable } from "./table";

const baseLog: LogEntry = {
  request_id: "req-1",
  api_key: "hashed-relay-key",
  team_id: "",
  model: "notion-ai",
  model_id: "",
  api_base: "https://www.notion.so",
  call_type: "litellm-relay",
  spend: 0,
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  startTime: "2026-07-09T22:36:30.141Z",
  endTime: "2026-07-09T22:36:30.220Z",
  custom_llm_provider: "",
  metadata: {
    app: "notion",
    status: "success",
    status_code: 200,
    user_api_key: null,
    user_api_key_alias: "relay-key",
    user_api_key_team_alias: null,
  },
  cache_hit: "False",
  cache_key: "",
  request_tags: { source: "notion" },
  messages: {},
  response: {},
  proxy_server_request: {},
  status: "success",
};

describe("view logs columns", () => {
  it("should render relay type with captured app logo and name", () => {
    render(<DataTable data={[baseLog]} columns={createColumns()} />);

    expect(screen.getByText("litellm-relay")).toBeInTheDocument();
    expect(screen.getAllByText("Notion").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("img", { name: "Notion logo" }).length).toBeGreaterThan(0);
    expect(screen.queryByRole("columnheader", { name: "Source" })).not.toBeInTheDocument();
    expect(screen.queryByText("LLM")).not.toBeInTheDocument();
  });

  it("should render collector rows with relay metadata as relay logs", () => {
    const legacyCollectorLog = {
      ...baseLog,
      request_id: "collector-01f159da",
      call_type: "completion",
      metadata: {
        ...baseLog.metadata,
        app: "codex",
        source: "litellm-relay",
      },
      model: "codex-ai",
      request_tags: ["litellm-relay", "codex"],
    };

    render(<DataTable data={[legacyCollectorLog]} columns={createColumns()} />);

    expect(screen.getByText("litellm-relay")).toBeInTheDocument();
    expect(screen.getAllByText("Codex").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("img", { name: "Codex logo" }).length).toBeGreaterThan(0);
    expect(screen.queryByText("LLM")).not.toBeInTheDocument();
  });

  it("should fall back to row api_key when relay metadata does not include a key hash", () => {
    render(<DataTable data={[baseLog]} columns={createColumns()} />);

    expect(screen.getByText("hashed-relay-key")).toBeInTheDocument();
  });
});

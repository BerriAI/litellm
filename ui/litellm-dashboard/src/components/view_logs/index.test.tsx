import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RequestViewer } from "./index";
import type { LogEntry } from "./columns";
import type { Row } from "@tanstack/react-table";

const baseLogEntry: LogEntry = {
  request_id: "chatcmpl-test-id",
  api_key: "api-key",
  team_id: "team-id",
  model: "gpt-4",
  model_id: "gpt-4",
  call_type: "chat",
  spend: 0,
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  startTime: "2025-11-14T00:00:00Z",
  endTime: "2025-11-14T00:00:00Z",
  cache_hit: "miss",
  duration: 1,
  messages: [{ role: "user", content: "hello" }],
  response: { status: "ok" },
  metadata: {
    status: "success",
    additional_usage_values: {
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  },
  request_tags: {},
  custom_llm_provider: "openai",
  api_base: "https://api.example.com",
};

const createRow = (overrides: Partial<LogEntry> = {}): Row<LogEntry> =>
  ({
    original: {
      ...baseLogEntry,
      ...overrides,
    },
  }) as unknown as Row<LogEntry>;

describe("Request Viewer", () => {
  it("renders the request details heading", () => {
    render(<RequestViewer row={createRow()} />);
    expect(screen.getByText("Request Details")).toBeInTheDocument();
  });

  it("should truncate the request id if it is longer than 64 characters", () => {
    const LONG_REQUEST_ID = "a".repeat(128);
    const TRUNCATED_REQUEST_ID = `${"a".repeat(64)}...`;
    render(
      <RequestViewer
        row={createRow({
          request_id: LONG_REQUEST_ID,
        })}
      />,
    );

    expect(screen.getByText(TRUNCATED_REQUEST_ID)).toBeInTheDocument();
  });

  it("should display LiteLLM Overhead when litellm_overhead_time_ms is present in metadata", () => {
    render(
      <RequestViewer
        row={createRow({
          metadata: {
            status: "success",
            litellm_overhead_time_ms: 150,
            additional_usage_values: {
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
          },
        })}
      />,
    );

    expect(screen.getByText("LiteLLM Overhead:")).toBeInTheDocument();
    expect(screen.getByText("150 ms")).toBeInTheDocument();
  });

  it("should not display LiteLLM Overhead when litellm_overhead_time_ms is not present in metadata", () => {
    render(<RequestViewer row={createRow()} />);

    expect(screen.queryByText("LiteLLM Overhead:")).not.toBeInTheDocument();
  });
});

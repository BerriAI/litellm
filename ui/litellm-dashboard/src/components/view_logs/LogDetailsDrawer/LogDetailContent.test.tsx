import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LogDetailContent } from "./LogDetailContent";
import type { LogEntry } from "../columns";

vi.mock("../GuardrailViewer/GuardrailViewer", () => ({
  default: ({ data }: { data: unknown }) => <div data-testid="guardrail-viewer">{JSON.stringify(data)}</div>,
}));

const createLogEntry = (overrides: Partial<LogEntry> = {}): LogEntry =>
  ({
    request_id: "chatcmpl-test-id",
    api_key: "api-key",
    team_id: "team-id",
    model: "gpt-4",
    model_id: "gpt-4",
    call_type: "chat",
    spend: 0,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    startTime: "2025-11-14T00:00:00Z",
    endTime: "2025-11-14T00:00:01Z",
    cache_hit: "miss",
    duration: 1,
    messages: [{ role: "user", content: "hello" }],
    response: { choices: [{ message: { content: "hi" } }] },
    metadata: { status: "success" },
    request_tags: {},
    custom_llm_provider: "openai",
    api_base: "https://api.example.com",
    ...overrides,
  }) as LogEntry;

describe("LogDetailContent", () => {
  it("should render the component successfully", () => {
    render(<LogDetailContent logEntry={createLogEntry()} />);

    expect(screen.getByText("Request Details")).toBeInTheDocument();
  });

  it("should display Request Details with model, provider, and call type", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          model: "gpt-4o",
          custom_llm_provider: "anthropic",
          call_type: "completion",
        })}
      />,
    );

    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("anthropic")).toBeInTheDocument();
    expect(screen.getByText("completion")).toBeInTheDocument();
  });

  it("should display error alert when request has failed", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          metadata: {
            status: "failure",
            error_information: {
              error_code: "rate_limit",
              error_message: "Too many requests",
              error_class: "RateLimitError",
            },
          },
        })}
      />,
    );

    expect(screen.getByText("Request Failed")).toBeInTheDocument();
    expect(screen.getByText("rate_limit")).toBeInTheDocument();
    expect(screen.getByText("Too many requests")).toBeInTheDocument();
  });

  it("should display tags section when request_tags has entries", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          request_tags: { env: "prod", version: "1.0" },
        })}
      />,
    );

    expect(screen.getByText("Tags")).toBeInTheDocument();
    expect(screen.getByText("env: prod")).toBeInTheDocument();
    expect(screen.getByText("version: 1.0")).toBeInTheDocument();
  });

  it("should not display tags section when request_tags is empty", () => {
    render(<LogDetailContent logEntry={createLogEntry({ request_tags: {} })} />);

    expect(screen.queryByText("Tags")).not.toBeInTheDocument();
  });

  it("should display Metrics section with tokens and cost", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
          spend: 0.002,
        })}
      />,
    );

    expect(screen.getByText("Metrics")).toBeInTheDocument();
    expect(screen.getAllByText("$0.00200000").length).toBeGreaterThanOrEqual(1);
  });

  it("should display ConfigInfoMessage when no messages, response, or error and not loading", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          messages: [],
          response: {},
          metadata: {},
        })}
      />,
    );

    expect(screen.getByText("Request/Response Data Not Available")).toBeInTheDocument();
  });

  it("should not display ConfigInfoMessage when isLoadingDetails is true even without data", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          messages: [],
          response: {},
          metadata: {},
        })}
        isLoadingDetails={true}
      />,
    );

    expect(screen.queryByText("Request/Response Data Not Available")).not.toBeInTheDocument();
  });

  it("should call onOpenSettings when user clicks open settings in ConfigInfoMessage", async () => {
    const onOpenSettings = vi.fn();
    const user = userEvent.setup();

    render(
      <LogDetailContent
        logEntry={createLogEntry({
          messages: [],
          response: {},
          metadata: {},
        })}
        onOpenSettings={onOpenSettings}
      />,
    );

    const settingsButton = screen.getByRole("button", { name: /open the settings/i });
    await user.click(settingsButton);

    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  it("should display loading state when isLoadingDetails is true", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry()}
        isLoadingDetails={true}
      />,
    );

    expect(screen.getByText("Loading request & response data...")).toBeInTheDocument();
  });

  it("should display Request & Response section with Pretty and JSON view modes", () => {
    render(<LogDetailContent logEntry={createLogEntry()} />);

    expect(screen.getByText("Request & Response")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Pretty" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "JSON" })).toBeInTheDocument();
  });

  it("should display Request and Response tabs when JSON view is selected", async () => {
    const user = userEvent.setup();
    render(<LogDetailContent logEntry={createLogEntry()} />);

    await user.click(screen.getByText("JSON"));

    expect(screen.getByRole("tab", { name: "Request" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Response" })).toBeInTheDocument();
  });

  it("should display response not available message when no response and Response tab is selected", async () => {
    const user = userEvent.setup();
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          response: {},
          metadata: { status: "success" },
        })}
      />,
    );

    await user.click(screen.getByText("JSON"));
    await user.click(screen.getByRole("tab", { name: "Response" }));

    expect(screen.getByText("Response data not available")).toBeInTheDocument();
  });

  it("should display Metadata section when metadata has keys", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          metadata: { status: "success", custom_key: "value" },
        })}
      />,
    );

    expect(screen.getByText("Metadata")).toBeInTheDocument();
  });

  it("should display IP address when requester_ip_address is present", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          requester_ip_address: "192.168.1.1",
        })}
      />,
    );

    expect(screen.getByText("192.168.1.1")).toBeInTheDocument();
  });

  it("should display guardrail label when guardrail data exists", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          metadata: {
            status: "success",
            guardrail_information: {
              guardrail_name: "PII Filter",
              masked_entity_count: { PERSON: 2 },
            },
          },
        })}
      />,
    );

    expect(screen.getByText("PII Filter")).toBeInTheDocument();
    expect(screen.getByText("2 masked")).toBeInTheDocument();
  });

  it("should display cache hit information when cache_hit is true", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          cache_hit: "true",
          metadata: {
            status: "success",
            additional_usage_values: {
              cache_read_input_tokens: 100,
              cache_creation_input_tokens: 0,
            },
          },
        })}
      />,
    );

    expect(screen.getByText("Cache Hit")).toBeInTheDocument();
    expect(screen.getByText("true")).toBeInTheDocument();
    expect(screen.getByText("Cache Read Tokens")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
  });

  it("should display LiteLLM Overhead when litellm_overhead_time_ms is in metadata", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          metadata: {
            status: "success",
            litellm_overhead_time_ms: 42.5,
          },
        })}
      />,
    );

    expect(screen.getByText("LiteLLM Overhead")).toBeInTheDocument();
    expect(screen.getByText("42.50 ms")).toBeInTheDocument();
  });

  it("should display start and end time in ISO format", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          startTime: "2025-11-14T12:00:00.000Z",
          endTime: "2025-11-14T12:00:01.500Z",
        })}
      />,
    );

    expect(screen.getByText("Start Time")).toBeInTheDocument();
    expect(screen.getByText("End Time")).toBeInTheDocument();
    const dateElements = screen.getAllByText((content) => content.includes("2025-11-14"));
    expect(dateElements.length).toBeGreaterThanOrEqual(2);
  });

  it("should display Vector Store Requests when vector store data exists", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          metadata: {
            status: "success",
            vector_store_request_metadata: [
              {
                query: "test query",
                vector_store_id: "vs-123",
                custom_llm_provider: "openai",
                start_time: 1700000000,
                end_time: 1700000001,
                vector_store_search_response: { data: [], search_query: "test" },
              },
            ],
          },
        })}
      />,
    );

    expect(screen.getByText("Vector Store Requests")).toBeInTheDocument();
  });

  it("should display provider as dash when custom_llm_provider is absent", () => {
    render(
      <LogDetailContent
        logEntry={createLogEntry({
          custom_llm_provider: undefined,
        })}
      />,
    );

    const descriptions = screen.getByText("Provider").closest(".ant-descriptions-item");
    expect(descriptions).toBeInTheDocument();
    expect(within(descriptions as HTMLElement).getByText("-")).toBeInTheDocument();
  });
});

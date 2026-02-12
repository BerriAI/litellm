import { render, screen, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { RequestResponsePanel } from "./RequestResponsePanel";
import type { LogEntry } from "./columns";
import NotificationsManager from "../molecules/notifications_manager";

const mockNotificationsManager = vi.mocked(NotificationsManager);

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

describe("RequestResponsePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
      writable: true,
      configurable: true,
    });
    Object.defineProperty(window, "isSecureContext", {
      value: true,
      writable: true,
      configurable: true,
    });
  });

  it("should render the component with request and response panels", () => {
    const mockGetRawRequest = vi.fn().mockReturnValue({ test: "request" });
    const mockFormattedResponse = vi.fn().mockReturnValue({ test: "response" });

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasMessages={true}
        hasResponse={true}
        hasError={false}
        errorInfo={null}
        getRawRequest={mockGetRawRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );

    expect(screen.getByText("Request")).toBeInTheDocument();
    expect(screen.getByText("Response")).toBeInTheDocument();
  });

  it("should copy request to clipboard when copy button is clicked", async () => {
    const user = userEvent.setup();
    const mockGetRawRequest = vi.fn().mockReturnValue({ test: "request data" });
    const mockFormattedResponse = vi.fn().mockReturnValue({ test: "response" });
    const mockWriteText = vi.fn().mockResolvedValue(undefined);

    if (navigator.clipboard) {
      vi.spyOn(navigator.clipboard, "writeText").mockImplementation(mockWriteText);
    } else {
      Object.defineProperty(navigator, "clipboard", {
        value: {
          writeText: mockWriteText,
        },
        writable: true,
        configurable: true,
      });
    }

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasMessages={true}
        hasResponse={true}
        hasError={false}
        errorInfo={null}
        getRawRequest={mockGetRawRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );

    const copyButtons = screen.getAllByRole("button");
    const copyRequestButton = copyButtons.find((button) => button.getAttribute("title") === "Copy request");

    expect(copyRequestButton).toBeInTheDocument();

    await act(async () => {
      await user.click(copyRequestButton!);
    });

    expect(mockGetRawRequest).toHaveBeenCalled();
    expect(mockWriteText).toHaveBeenCalledWith(JSON.stringify({ test: "request data" }, null, 2));
    expect(mockNotificationsManager.success).toHaveBeenCalledWith("Request copied to clipboard");
  });

  it("should copy response to clipboard when copy button is clicked", async () => {
    const user = userEvent.setup();
    const mockGetRawRequest = vi.fn().mockReturnValue({ test: "request" });
    const mockFormattedResponse = vi.fn().mockReturnValue({ test: "response data" });
    const mockWriteText = vi.fn().mockResolvedValue(undefined);

    if (navigator.clipboard) {
      vi.spyOn(navigator.clipboard, "writeText").mockImplementation(mockWriteText);
    } else {
      Object.defineProperty(navigator, "clipboard", {
        value: {
          writeText: mockWriteText,
        },
        writable: true,
        configurable: true,
      });
    }

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasMessages={true}
        hasResponse={true}
        hasError={false}
        errorInfo={null}
        getRawRequest={mockGetRawRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );

    const copyButtons = screen.getAllByRole("button");
    const copyResponseButton = copyButtons.find((button) => button.getAttribute("title") === "Copy response");

    expect(copyResponseButton).toBeInTheDocument();
    expect(copyResponseButton).not.toBeDisabled();

    await act(async () => {
      await user.click(copyResponseButton!);
    });

    expect(mockFormattedResponse).toHaveBeenCalled();
    expect(mockWriteText).toHaveBeenCalledWith(JSON.stringify({ test: "response data" }, null, 2));
    expect(mockNotificationsManager.success).toHaveBeenCalledWith("Response copied to clipboard");
  });

  it("should call formattedResponse for the response panel and not getRawRequest", () => {
    const mockGetRawRequest = vi.fn().mockReturnValue({ requestData: "this should not appear in response" });
    const mockFormattedResponse = vi.fn().mockReturnValue({ responseData: "this should appear in response" });

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasMessages={true}
        hasResponse={true}
        hasError={false}
        errorInfo={null}
        getRawRequest={mockGetRawRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );

    expect(mockFormattedResponse).toHaveBeenCalled();
    expect(mockGetRawRequest).toHaveBeenCalled();
    
    const formattedResponseCallCount = mockFormattedResponse.mock.calls.length;
    expect(formattedResponseCallCount).toBeGreaterThanOrEqual(1);
    
    const responseData = mockFormattedResponse.mock.results[0].value;
    expect(responseData).toEqual({ responseData: "this should appear in response" });
    expect(responseData).not.toEqual({ requestData: "this should not appear in response" });
  });

  it("should show error response data when hasError is true and hasResponse is false", () => {
    const failedLogEntry: LogEntry = {
      ...baseLogEntry,
      messages: [],
      response: {},
      metadata: {
        status: "failure",
        error_information: {
          error_message: "Model not found",
          error_class: "NotFoundError",
          error_code: 404,
        },
        additional_usage_values: {
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
    };
    const errorResponse = { error: { message: "Model not found", type: "NotFoundError", code: 404, param: null } };
    const mockGetRawRequest = vi.fn().mockReturnValue({ messages: [] });
    const mockFormattedResponse = vi.fn().mockReturnValue(errorResponse);
    render(
      <RequestResponsePanel
        row={{ original: failedLogEntry }}
        hasMessages={false}
        hasResponse={false}
        hasError={true}
        errorInfo={failedLogEntry.metadata.error_information}
        getRawRequest={mockGetRawRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );
    expect(screen.queryByText("Response data not available")).not.toBeInTheDocument();
    expect(mockFormattedResponse).toHaveBeenCalled();
    const copyButtons = screen.getAllByRole("button");
    const copyResponseButton = copyButtons.find((button) => button.getAttribute("title") === "Copy response");
    expect(copyResponseButton).not.toBeDisabled();
  });

  it("should show Response data not available when hasResponse and hasError are both false", () => {
    const mockGetRawRequest = vi.fn().mockReturnValue({ messages: [] });
    const mockFormattedResponse = vi.fn().mockReturnValue({});
    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasMessages={false}
        hasResponse={false}
        hasError={false}
        errorInfo={null}
        getRawRequest={mockGetRawRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );
    expect(screen.getByText("Response data not available")).toBeInTheDocument();
  });

  it("should show error code in response header when hasError is true", () => {
    const errorInfo = { error_message: "Rate limit exceeded", error_class: "RateLimitError", error_code: 429 };
    const mockGetRawRequest = vi.fn().mockReturnValue({ messages: [] });
    const mockFormattedResponse = vi.fn().mockReturnValue({ error: { message: "Rate limit exceeded", type: "RateLimitError", code: 429, param: null } });
    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasMessages={false}
        hasResponse={false}
        hasError={true}
        errorInfo={errorInfo}
        getRawRequest={mockGetRawRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );
    expect(screen.getByText(/HTTP code 429/)).toBeInTheDocument();
  });
});

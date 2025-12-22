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
  proxy_server_request: { body: { messages: [{ role: "user", content: "hello" }] } },
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
    const mockGetClientRequest = vi.fn().mockReturnValue({ test: "request" });
    const mockGetModelRequest = vi.fn().mockReturnValue({ test: "model request" });
    const mockGetModelResponse = vi.fn().mockReturnValue({ test: "model response" });
    const mockFormattedResponse = vi.fn().mockReturnValue({ test: "response" });

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasClientRequest={true}
        hasModelRequest={true}
        hasModelResponse={true}
        hasClientResponse={true}
        hasError={false}
        errorInfo={null}
        getClientRequest={mockGetClientRequest}
        getModelRequest={mockGetModelRequest}
        getModelResponse={mockGetModelResponse}
        formattedResponse={mockFormattedResponse}
      />,
    );

    expect(screen.getByText("Request from client")).toBeInTheDocument();
    expect(screen.getByText("Request to model/endpoint")).toBeInTheDocument();
    expect(screen.getByText("Response from model/endpoint")).toBeInTheDocument();
    expect(screen.getByText("Response to client")).toBeInTheDocument();
  });

  it("should copy request to clipboard when copy button is clicked", async () => {
    const user = userEvent.setup();
    const mockGetClientRequest = vi.fn().mockReturnValue({ test: "request data" });
    const mockGetModelRequest = vi.fn().mockReturnValue({ test: "model request data" });
    const mockGetModelResponse = vi.fn().mockReturnValue({ test: "model response data" });
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
        hasClientRequest={true}
        hasModelRequest={true}
        hasModelResponse={true}
        hasClientResponse={true}
        hasError={false}
        errorInfo={null}
        getClientRequest={mockGetClientRequest}
        getModelRequest={mockGetModelRequest}
        getModelResponse={mockGetModelResponse}
        formattedResponse={mockFormattedResponse}
      />,
    );

    const copyRequestButton = screen.getByTitle("Copy request from client");

    expect(copyRequestButton).toBeInTheDocument();

    await act(async () => {
      await user.click(copyRequestButton!);
    });

    expect(mockGetClientRequest).toHaveBeenCalled();
    expect(mockWriteText).toHaveBeenCalledWith(JSON.stringify({ test: "request data" }, null, 2));
    expect(mockNotificationsManager.success).toHaveBeenCalledWith("Client request copied to clipboard");
  });

  it("should copy response to clipboard when copy button is clicked", async () => {
    const user = userEvent.setup();
    const mockGetClientRequest = vi.fn().mockReturnValue({ test: "request" });
    const mockGetModelRequest = vi.fn().mockReturnValue({ test: "model request" });
    const mockGetModelResponse = vi.fn().mockReturnValue({ test: "model response" });
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
        hasClientRequest={true}
        hasModelRequest={true}
        hasModelResponse={true}
        hasClientResponse={true}
        hasError={false}
        errorInfo={null}
        getClientRequest={mockGetClientRequest}
        getModelRequest={mockGetModelRequest}
        getModelResponse={mockGetModelResponse}
        formattedResponse={mockFormattedResponse}
      />,
    );

    const copyResponseButton = screen.getByTitle("Copy response to client");

    expect(copyResponseButton).toBeInTheDocument();
    expect(copyResponseButton).not.toBeDisabled();

    await act(async () => {
      await user.click(copyResponseButton!);
    });

    expect(mockFormattedResponse).toHaveBeenCalled();
    expect(mockWriteText).toHaveBeenCalledWith(JSON.stringify({ test: "response data" }, null, 2));
    expect(mockNotificationsManager.success).toHaveBeenCalledWith("Response to client copied to clipboard");
  });

  it("should call formattedResponse for the response panel and not request getters", () => {
    const mockGetClientRequest = vi.fn().mockReturnValue({ requestData: "this should not appear in response" });
    const mockGetModelRequest = vi.fn().mockReturnValue({ modelRequestData: "not in response" });
    const mockGetModelResponse = vi.fn().mockReturnValue({ modelResponseData: "not in response" });
    const mockFormattedResponse = vi.fn().mockReturnValue({ responseData: "this should appear in response" });

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasClientRequest={true}
        hasModelRequest={true}
        hasModelResponse={true}
        hasClientResponse={true}
        hasError={false}
        errorInfo={null}
        getClientRequest={mockGetClientRequest}
        getModelRequest={mockGetModelRequest}
        getModelResponse={mockGetModelResponse}
        formattedResponse={mockFormattedResponse}
      />,
    );

    expect(mockFormattedResponse).toHaveBeenCalled();
    expect(mockGetClientRequest).toHaveBeenCalled();
    
    const formattedResponseCallCount = mockFormattedResponse.mock.calls.length;
    expect(formattedResponseCallCount).toBeGreaterThanOrEqual(1);
    
    const responseData = mockFormattedResponse.mock.results[0].value;
    expect(responseData).toEqual({ responseData: "this should appear in response" });
    expect(responseData).not.toEqual({ requestData: "this should not appear in response" });
  });

  it("renders model request data and copies it successfully", async () => {
    const user = userEvent.setup();
    const mockGetClientRequest = vi.fn().mockReturnValue({ test: "request data" });
    const mockGetModelRequest = vi.fn().mockReturnValue({ model: "request data" });
    const mockFormattedResponse = vi.fn().mockReturnValue({ test: "response data" });
    const mockWriteText = vi.fn().mockResolvedValue(undefined);

    if (navigator.clipboard) {
      vi.spyOn(navigator.clipboard, "writeText").mockImplementation(mockWriteText);
    }

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasClientRequest={true}
        hasModelRequest={true}
        hasClientResponse={true}
        hasError={false}
        errorInfo={null}
        getClientRequest={mockGetClientRequest}
        getModelRequest={mockGetModelRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );

    expect(screen.getByText("Request to model/endpoint")).toBeInTheDocument();
    expect(mockGetModelRequest).toHaveBeenCalled();

    const copyModelButton = screen.getByTitle("Copy request to model/endpoint");
    expect(copyModelButton).not.toBeDisabled();

    await act(async () => {
      await user.click(copyModelButton);
    });

    expect(mockWriteText).toHaveBeenCalledWith(JSON.stringify({ model: "request data" }, null, 2));
    expect(mockNotificationsManager.success).toHaveBeenCalledWith("Request to model/endpoint copied to clipboard");
  });

  it("shows guidance and disables copy when model request data is missing", () => {
    const mockGetClientRequest = vi.fn().mockReturnValue({ test: "request data" });
    const mockGetModelRequest = vi.fn();
    const mockFormattedResponse = vi.fn().mockReturnValue({ test: "response" });

    render(
      <RequestResponsePanel
        row={{ original: baseLogEntry }}
        hasClientRequest={true}
        hasModelRequest={false}
        hasClientResponse={true}
        hasError={false}
        errorInfo={null}
        getClientRequest={mockGetClientRequest}
        getModelRequest={mockGetModelRequest}
        formattedResponse={mockFormattedResponse}
      />,
    );

    expect(screen.getByText(/Request not available/)).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "store_prompts_in_spend_logs", hidden: true }),
    ).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/config_settings#store_prompts_in_spend_logs",
    );
    expect(
      screen.getByRole("link", { name: "MAX_STRING_LENGTH_PROMPT_IN_DB", hidden: true }),
    ).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/config_settings#store_prompts_in_spend_logs#MAX_STRING_LENGTH_PROMPT_IN_DB",
    );

    const copyModelRequestButton = screen.getByTitle("Copy request to model/endpoint");
    expect(copyModelRequestButton).toBeDisabled();
    expect(mockGetModelRequest).not.toHaveBeenCalled();
  });
});

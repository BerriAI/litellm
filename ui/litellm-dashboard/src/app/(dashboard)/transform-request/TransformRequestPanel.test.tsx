import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TransformRequestPanel from "./TransformRequestPanel";
import { transformRequestCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";

vi.mock("@/components/networking", () => ({
  transformRequestCall: vi.fn(),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    info: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const transformRequestCallMock = vi.mocked(transformRequestCall);
const notify = vi.mocked(NotificationsManager);

const ACCESS_TOKEN = "sk-test-token";

const getRequestTextarea = () => screen.getByPlaceholderText(/press cmd\/ctrl \+ enter to transform/i);

const getTransformButton = () => screen.getByRole("button", { name: /transform/i });

const getCopyButton = () => screen.getByRole("button", { name: /copy to clipboard/i });

describe("TransformRequestPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders both panels, the prefilled request and the placeholder curl", () => {
    render(<TransformRequestPanel accessToken={ACCESS_TOKEN} />);

    expect(screen.getByText("Original Request")).toBeInTheDocument();
    expect(screen.getByText("Transformed Request")).toBeInTheDocument();
    expect(screen.getByText(/sensitive headers are not shown/i)).toBeInTheDocument();

    expect((getRequestTextarea() as HTMLTextAreaElement).value).toContain('"model": "openai/gpt-4o"');
    expect(screen.getByText(/https:\/\/api\.openai\.com\/v1\/chat\/completions/)).toBeInTheDocument();

    expect(screen.getByRole("link", { name: /here/i })).toHaveAttribute(
      "href",
      "https://github.com/BerriAI/litellm/issues",
    );
  });

  it("sends the edited request body as a completion call and renders the returned curl", async () => {
    const user = userEvent.setup();
    transformRequestCallMock.mockResolvedValue({
      raw_request_api_base: "https://api.anthropic.com/v1/messages",
      raw_request_body: { model: "claude-opus-4-8", max_tokens: 42 },
      raw_request_headers: { "x-api-key": "redacted" },
    });

    render(<TransformRequestPanel accessToken={ACCESS_TOKEN} />);

    const textarea = getRequestTextarea();
    await user.clear(textarea);
    await user.type(textarea, '{{"model": "claude-opus-4-8"}');

    await user.click(getTransformButton());

    await waitFor(() => expect(transformRequestCallMock).toHaveBeenCalledTimes(1));
    expect(transformRequestCallMock).toHaveBeenCalledWith(ACCESS_TOKEN, {
      call_type: "completion",
      request_body: { model: "claude-opus-4-8" },
    });

    const output = await screen.findByText(/api\.anthropic\.com\/v1\/messages/);
    expect(output.textContent).toContain("curl -X POST");
    expect(output.textContent).toContain("-H 'x-api-key: redacted'");
    expect(output.textContent).toContain('"model": "claude-opus-4-8"');
    expect(output.textContent).toContain('"max_tokens": 42');
    expect(notify.success).toHaveBeenCalledWith("Request transformed successfully");
  });

  it("transforms on Cmd/Ctrl + Enter without clicking the button", async () => {
    const user = userEvent.setup();
    transformRequestCallMock.mockResolvedValue({
      raw_request_api_base: "https://api.openai.com/v1/chat/completions",
      raw_request_body: { model: "gpt-4o" },
      raw_request_headers: {},
    });

    render(<TransformRequestPanel accessToken={ACCESS_TOKEN} />);

    getRequestTextarea().focus();
    await user.keyboard("{Meta>}{Enter}{/Meta}");

    await waitFor(() => expect(transformRequestCallMock).toHaveBeenCalledTimes(1));
  });

  it("rejects invalid JSON without calling the backend", async () => {
    const user = userEvent.setup();

    render(<TransformRequestPanel accessToken={ACCESS_TOKEN} />);

    const textarea = getRequestTextarea();
    await user.clear(textarea);
    await user.type(textarea, "not json");
    await user.click(getTransformButton());

    await waitFor(() => expect(notify.fromBackend).toHaveBeenCalledWith("Invalid JSON in request body"));
    expect(transformRequestCallMock).not.toHaveBeenCalled();
  });

  it("does not call the backend when there is no access token", async () => {
    const user = userEvent.setup();

    render(<TransformRequestPanel accessToken={null} />);

    await user.click(getTransformButton());

    await waitFor(() => expect(notify.fromBackend).toHaveBeenCalledWith("No access token found"));
    expect(transformRequestCallMock).not.toHaveBeenCalled();
  });

  it("reports a failed transform and leaves the placeholder curl in place", async () => {
    const user = userEvent.setup();
    vi.spyOn(console, "error").mockImplementation(() => {});
    transformRequestCallMock.mockRejectedValue(new Error("boom"));

    render(<TransformRequestPanel accessToken={ACCESS_TOKEN} />);

    await user.click(getTransformButton());

    await waitFor(() => expect(notify.fromBackend).toHaveBeenCalledWith("Failed to transform request"));
    expect(screen.getByText(/https:\/\/api\.openai\.com\/v1\/chat\/completions/)).toBeInTheDocument();
  });

  it("copies the transformed request to the clipboard", async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText");
    transformRequestCallMock.mockResolvedValue({
      raw_request_api_base: "https://api.anthropic.com/v1/messages",
      raw_request_body: { model: "claude-opus-4-8" },
      raw_request_headers: {},
    });

    render(<TransformRequestPanel accessToken={ACCESS_TOKEN} />);

    await user.click(getTransformButton());
    await screen.findByText(/api\.anthropic\.com\/v1\/messages/);

    await user.click(getCopyButton());

    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0]?.[0]).toContain("https://api.anthropic.com/v1/messages");
    expect(notify.success).toHaveBeenCalledWith("Copied to clipboard");
  });
});

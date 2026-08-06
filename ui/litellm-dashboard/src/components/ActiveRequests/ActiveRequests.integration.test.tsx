import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ActiveRequests from "./ActiveRequests";
import { activeRequestsCall, cancelActiveRequestCall, type ActiveRequestsResponse } from "./activeRequestsApi";

const mockReplace = vi.fn();
const searchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () => searchParams,
}));

vi.mock("./activeRequestsApi", () => ({
  activeRequestsCall: vi.fn(),
  cancelActiveRequestCall: vi.fn(),
}));

const mockedActiveRequestsCall = vi.mocked(activeRequestsCall);
const mockedCancelCall = vi.mocked(cancelActiveRequestCall);

const page: ActiveRequestsResponse = {
  available: true,
  items: [
    {
      registry_id: "reg-123",
      request_id: "call-123",
      started_at: Date.now() / 1000 - 12,
      model: "model-a",
      streaming: true,
      end_user_id: "end-user-123",
      user_id: "user-1",
      user_email: "user@example.test",
      organization_id: "org-1",
      project_id: "project-1",
      team_alias: "AI Team",
      key_alias: "production",
      route: "/v1/chat/completions",
      pod: "proxy-1",
    },
  ],
  total: 1,
  page: 1,
  page_size: 50,
  truncated: false,
  reason: null,
  generated_at: new Date().toISOString(),
};

describe("ActiveRequests", () => {
  beforeEach(() => {
    mockedActiveRequestsCall.mockReset();
    mockedCancelCall.mockReset();
    mockReplace.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should render request ownership returned by the live registry", async () => {
    mockedActiveRequestsCall.mockResolvedValue(page);

    render(<ActiveRequests accessToken="token" />);

    await waitFor(() => expect(screen.getByText("end-user-123")).toBeInTheDocument());
    expect(screen.getByText("org-1")).toBeInTheDocument();
    expect(screen.getByText("project-1")).toBeInTheDocument();
    expect(screen.getByText("user@example.test")).toBeInTheDocument();
  });

  it("should show a safe error when refreshing fails", async () => {
    mockedActiveRequestsCall.mockRejectedValue(new Error("Registry unavailable"));

    render(<ActiveRequests accessToken="token" />);

    await waitFor(() => expect(screen.getByText("Could not refresh active requests")).toBeInTheDocument());
    expect(screen.getByText("Registry unavailable")).toBeInTheDocument();
  });

  it("should not overlap polling requests when the proxy is slow", async () => {
    vi.useFakeTimers();
    mockedActiveRequestsCall.mockImplementation(() => new Promise(() => {}));

    const { unmount } = render(<ActiveRequests accessToken="token" />);
    await act(async () => vi.advanceTimersByTime(0));
    expect(mockedActiveRequestsCall).toHaveBeenCalledTimes(1);

    await act(async () => vi.advanceTimersByTime(15000));
    expect(mockedActiveRequestsCall).toHaveBeenCalledTimes(1);
    unmount();
  });

  it("should cancel by registry id, not by request id", async () => {
    mockedActiveRequestsCall.mockResolvedValue(page);
    mockedCancelCall.mockResolvedValue({ cancelled: true, detail: "Cancellation sent to the worker" });

    render(<ActiveRequests accessToken="token" />);
    await waitFor(() => expect(screen.getByText("end-user-123")).toBeInTheDocument());

    fireEvent.click(screen.getByText("end-user-123"));
    const panel = await screen.findByRole("dialog");
    fireEvent.click(screen.getByRole("button", { name: "Cancel request" }));

    await waitFor(() => expect(mockedCancelCall).toHaveBeenCalledWith("reg-123"));
    expect(panel).not.toBeInTheDocument();
  });

  it("should surface a failed cancellation instead of closing silently", async () => {
    mockedActiveRequestsCall.mockResolvedValue(page);
    mockedCancelCall.mockRejectedValue(new Error("That request is no longer running"));

    render(<ActiveRequests accessToken="token" />);
    await waitFor(() => expect(screen.getByText("end-user-123")).toBeInTheDocument());

    fireEvent.click(screen.getByText("end-user-123"));
    await screen.findByRole("dialog");
    fireEvent.click(screen.getByRole("button", { name: "Cancel request" }));

    await waitFor(() => expect(screen.getByText("That request is no longer running")).toBeInTheDocument());
  });

  it("should stop polling while paused and resume afterwards", async () => {
    vi.useFakeTimers();
    mockedActiveRequestsCall.mockResolvedValue(page);

    render(<ActiveRequests accessToken="token" />);
    await act(async () => vi.advanceTimersByTime(0));
    expect(mockedActiveRequestsCall).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("switch"));
    await act(async () => vi.advanceTimersByTime(20000));
    expect(mockedActiveRequestsCall).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("switch"));
    await act(async () => vi.advanceTimersByTime(6000));
    expect(mockedActiveRequestsCall.mock.calls.length).toBeGreaterThan(1);
  });

  it("should put the active filters in the url so the view can be shared", async () => {
    vi.useFakeTimers();
    mockedActiveRequestsCall.mockResolvedValue(page);

    render(<ActiveRequests accessToken="token" />);
    await act(async () => vi.advanceTimersByTime(0));

    fireEvent.change(screen.getByPlaceholderText("End User ID"), { target: { value: "end-user-9" } });
    await act(async () => vi.advanceTimersByTime(400));

    expect(mockReplace).toHaveBeenCalledWith("?end_user_id=end-user-9", { scroll: false });
  });
});

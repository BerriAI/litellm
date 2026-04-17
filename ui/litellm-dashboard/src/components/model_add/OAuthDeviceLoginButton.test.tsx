import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import OAuthDeviceLoginButton from "./OAuthDeviceLoginButton";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token" }),
}));

describe("OAuthDeviceLoginButton", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  const baseProps = {
    providerLabel: "ChatGPT",
    startCall: vi.fn(),
    statusCall: vi.fn(),
    cancelCall: vi.fn(),
    onSuccess: vi.fn(),
  };

  it("disables Sign-in when no credential name is provided", () => {
    render(<OAuthDeviceLoginButton {...baseProps} onSuccess={vi.fn()} />);
    const button = screen.getByRole("button", { name: /Sign in with ChatGPT/i });
    expect(button).toBeDisabled();
  });

  it("enables Sign-in once a credential name is provided", () => {
    render(
      <OAuthDeviceLoginButton
        {...baseProps}
        credentialName="my-creds"
        onSuccess={vi.fn()}
      />,
    );
    expect(
      screen.getByRole("button", { name: /Sign in with ChatGPT/i }),
    ).toBeEnabled();
  });

  it("on Sign-in click, shows the user code and verification URL returned by startCall", async () => {
    const startCall = vi.fn().mockResolvedValue({
      session_id: "sess-1",
      user_code: "ABCD-1234",
      verification_url: "https://auth.example/code",
      interval: 5,
    });
    const statusCall = vi.fn().mockResolvedValue({ status: "pending" });

    render(
      <OAuthDeviceLoginButton
        {...baseProps}
        credentialName="my-creds"
        startCall={startCall}
        statusCall={statusCall}
        cancelCall={vi.fn()}
        onSuccess={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Sign in with ChatGPT/i }));

    await waitFor(() => {
      expect(screen.getByText("ABCD-1234")).toBeInTheDocument();
    });
    expect(
      screen.getByRole("link", { name: "https://auth.example/code" }),
    ).toHaveAttribute("href", "https://auth.example/code");
    expect(startCall).toHaveBeenCalledWith("test-token", "my-creds");
  });

  it("calls onSuccess when polling reports success", async () => {
    const onSuccess = vi.fn();
    const startCall = vi.fn().mockResolvedValue({
      session_id: "sess-1",
      user_code: "CODE",
      verification_url: "https://auth.example/code",
      interval: 5,
    });
    const statusCall = vi.fn().mockResolvedValue({ status: "success" });

    render(
      <OAuthDeviceLoginButton
        {...baseProps}
        credentialName="my-creds"
        startCall={startCall}
        statusCall={statusCall}
        cancelCall={vi.fn()}
        onSuccess={onSuccess}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Sign in with ChatGPT/i }));

    // Poll interval is 3s; allow up to 10s of real time for the first poll
    // to fire and resolve with status=success.
    await waitFor(
      () => {
        expect(onSuccess).toHaveBeenCalledTimes(1);
      },
      { timeout: 10000 },
    );
  }, 15000);

  it("surfaces error status from polling", async () => {
    const startCall = vi.fn().mockResolvedValue({
      session_id: "sess-1",
      user_code: "CODE",
      verification_url: "https://auth.example/code",
      interval: 5,
    });
    const statusCall = vi
      .fn()
      .mockResolvedValue({ status: "error", message: "upstream failure" });

    render(
      <OAuthDeviceLoginButton
        {...baseProps}
        credentialName="my-creds"
        startCall={startCall}
        statusCall={statusCall}
        cancelCall={vi.fn()}
        onSuccess={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Sign in with ChatGPT/i }));

    await waitFor(
      () => {
        expect(screen.getByText("upstream failure")).toBeInTheDocument();
      },
      { timeout: 10000 },
    );
  }, 15000);

  it("on Cancel click, calls cancelCall and returns to idle state", async () => {
    const cancelCall = vi.fn().mockResolvedValue(undefined);
    const startCall = vi.fn().mockResolvedValue({
      session_id: "sess-1",
      user_code: "CODE",
      verification_url: "https://auth.example/code",
      interval: 5,
    });
    const statusCall = vi.fn().mockResolvedValue({ status: "pending" });

    render(
      <OAuthDeviceLoginButton
        {...baseProps}
        credentialName="my-creds"
        startCall={startCall}
        statusCall={statusCall}
        cancelCall={cancelCall}
        onSuccess={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Sign in with ChatGPT/i }));
    await waitFor(() => {
      expect(screen.getByText("CODE")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /Cancel/i }));

    await waitFor(() => {
      expect(cancelCall).toHaveBeenCalledWith("test-token", "sess-1");
    });
    // Back to idle — Sign-in button visible again
    expect(
      screen.getByRole("button", { name: /Sign in with ChatGPT/i }),
    ).toBeInTheDocument();
  });
});

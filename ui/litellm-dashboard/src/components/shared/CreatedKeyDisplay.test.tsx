import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CreatedKeyDisplay from "./CreatedKeyDisplay";

vi.mock("@/components/molecules/message_manager", () => ({
  default: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn(), loading: vi.fn(), destroy: vi.fn() },
}));

vi.mock("@/components/networking", () => ({
  keyShareCreateCall: vi.fn(),
}));

import MessageManager from "@/components/molecules/message_manager";
import { keyShareCreateCall } from "@/components/networking";

describe("CreatedKeyDisplay", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should render", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByText("sk-test-123")).toBeInTheDocument();
  });

  it("should display the security warning", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByText(/you will not be able to view it again/i)).toBeInTheDocument();
  });

  it("should show the copy button with initial label", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByRole("button", { name: /copy virtual key/i })).toBeInTheDocument();
  });

  it("should change button text to Copied after clicking copy", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));

    expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();
  });

  it("should show a success message when the key is copied", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));

    expect(MessageManager.success).toHaveBeenCalledWith("Key copied to clipboard");
  });

  it("should revert button text back after 2 seconds", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));
    expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(screen.getByRole("button", { name: /copy virtual key/i })).toBeInTheDocument();
  });

  it("should not show the Create secure share link button without an accessToken", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.queryByRole("button", { name: /create secure share link/i })).not.toBeInTheDocument();
  });

  it("should show the Create secure share link button when an accessToken is provided", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" accessToken="sk-admin" />);
    expect(screen.getByRole("button", { name: /create secure share link/i })).toBeInTheDocument();
  });

  it("should render the returned share link after clicking Create secure share link", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const shareResponse = {
      share_link: "http://localhost:4000/key/share/tok_abc123",
      token: "tok_abc123",
      expires_at: "2026-01-01T00:00:00Z",
      one_time_only: true,
    };
    vi.mocked(keyShareCreateCall).mockResolvedValue(shareResponse);

    render(<CreatedKeyDisplay apiKey="sk-test-123" accessToken="sk-admin" />);

    await user.click(screen.getByRole("button", { name: /create secure share link/i }));

    expect(keyShareCreateCall).toHaveBeenCalledWith("sk-admin", "sk-test-123", {
      expire_after: "OneDay",
    });
    const link = await screen.findByRole("link", { name: /key\/share\/tok_abc123/i });
    expect(link).toHaveAttribute("href", "http://localhost:4000/key/share/tok_abc123");
    expect(screen.getByRole("button", { name: /copy share link/i })).toBeInTheDocument();
  });

  it("should surface an error when share link creation fails", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(keyShareCreateCall).mockRejectedValue(new Error("Key not found"));

    render(<CreatedKeyDisplay apiKey="sk-test-123" accessToken="sk-admin" />);

    await user.click(screen.getByRole("button", { name: /create secure share link/i }));

    await vi.waitFor(() => {
      expect(MessageManager.error).toHaveBeenCalledWith(expect.stringContaining("Key not found"));
    });
  });
});

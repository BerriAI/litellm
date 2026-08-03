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

  it("should not show the share button when no accessToken is provided", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.queryByRole("button", { name: /securely share/i })).not.toBeInTheDocument();
  });

  it("should show the share button when an accessToken is provided", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" accessToken="sk-admin" />);
    expect(screen.getByRole("button", { name: /securely share/i })).toBeInTheDocument();
  });

  it("should create and display a share link when the share button is clicked", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(keyShareCreateCall).mockResolvedValue({
      share_link: "https://password.link/abc/#pub",
    });

    render(<CreatedKeyDisplay apiKey="sk-test-123" accessToken="sk-admin" />);
    await user.click(screen.getByRole("button", { name: /securely share/i }));

    expect(keyShareCreateCall).toHaveBeenCalledWith("sk-admin", "sk-test-123");
    expect(await screen.findByText("https://password.link/abc/#pub")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /copy share link/i })).toBeInTheDocument();
  });

  it("should surface an error and show no link when the share call fails", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(keyShareCreateCall).mockRejectedValue(new Error("boom"));

    render(<CreatedKeyDisplay apiKey="sk-test-123" accessToken="sk-admin" />);
    await user.click(screen.getByRole("button", { name: /securely share/i }));

    expect(keyShareCreateCall).toHaveBeenCalled();
    expect(vi.mocked(MessageManager.error)).toHaveBeenCalledWith("Failed to create secure share link. boom");
    expect(screen.queryByRole("button", { name: /copy share link/i })).not.toBeInTheDocument();
  });
});

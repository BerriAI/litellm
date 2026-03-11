import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import BlockToggle from "./BlockToggle";
import { message } from "antd";

// Mock antd message
vi.mock("antd", async () => {
  const actual = await vi.importActual("antd");
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
    },
  };
});

// Mock fetch
global.fetch = vi.fn();

describe("BlockToggle", () => {
  const defaultProps = {
    entityType: "user" as const,
    entityId: "test-user-123",
    currentBlockedStatus: false,
    accessToken: "test-token",
    baseUrl: "http://localhost:4000",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (global.fetch as any).mockReset();
  });

  it("should render with active status when not blocked", () => {
    render(<BlockToggle {...defaultProps} />);
    
    const switchElement = screen.getByRole("switch");
    expect(switchElement).toBeInTheDocument();
    expect(switchElement).toHaveAttribute("aria-checked", "true");
  });

  it("should render with blocked status and badge when blocked", () => {
    render(<BlockToggle {...defaultProps} currentBlockedStatus={true} />);
    
    const switchElement = screen.getByRole("switch");
    expect(switchElement).toHaveAttribute("aria-checked", "false");
    expect(screen.getByText("BLOCKED")).toBeInTheDocument();
  });

  it("should call block API when toggling from active to blocked", async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ blocked: true }),
    });

    render(<BlockToggle {...defaultProps} />);
    
    const switchElement = screen.getByRole("switch");
    fireEvent.click(switchElement);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:4000/user/block",
        expect.objectContaining({
          method: "POST",
          headers: {
            Authorization: "Bearer test-token",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ user_id: "test-user-123" }),
        })
      );
    });

    expect(message.success).toHaveBeenCalledWith("User blocked successfully");
  });

  it("should call unblock API when toggling from blocked to active", async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ blocked: false }),
    });

    render(<BlockToggle {...defaultProps} currentBlockedStatus={true} />);
    
    const switchElement = screen.getByRole("switch");
    fireEvent.click(switchElement);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:4000/user/unblock",
        expect.objectContaining({
          method: "POST",
          headers: {
            Authorization: "Bearer test-token",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ user_id: "test-user-123" }),
        })
      );
    });

    expect(message.success).toHaveBeenCalledWith("User unblocked successfully");
  });

  it("should handle API errors gracefully", async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: "Permission denied" }),
    });

    render(<BlockToggle {...defaultProps} />);
    
    const switchElement = screen.getByRole("switch");
    fireEvent.click(switchElement);

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith("Permission denied");
    });
  });

  it("should call onToggle callback when provided", async () => {
    const onToggleMock = vi.fn().mockResolvedValue(undefined);
    
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ blocked: true }),
    });

    render(<BlockToggle {...defaultProps} onToggle={onToggleMock} />);
    
    const switchElement = screen.getByRole("switch");
    fireEvent.click(switchElement);

    await waitFor(() => {
      expect(onToggleMock).toHaveBeenCalledWith(true);
    });
  });

  it("should work with team entity type", async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ blocked: true }),
    });

    render(
      <BlockToggle
        {...defaultProps}
        entityType="team"
        entityId="test-team-123"
      />
    );
    
    const switchElement = screen.getByRole("switch");
    fireEvent.click(switchElement);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:4000/team/block",
        expect.objectContaining({
          body: JSON.stringify({ team_id: "test-team-123" }),
        })
      );
    });

    expect(message.success).toHaveBeenCalledWith("Team blocked successfully");
  });

  it("should work with key entity type", async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ blocked: true }),
    });

    render(
      <BlockToggle
        {...defaultProps}
        entityType="key"
        entityId="test-key-123"
      />
    );
    
    const switchElement = screen.getByRole("switch");
    fireEvent.click(switchElement);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:4000/key/block",
        expect.objectContaining({
          body: JSON.stringify({ key: "test-key-123" }),
        })
      );
    });

    expect(message.success).toHaveBeenCalledWith("Key blocked successfully");
  });

  it("should show loading state during API call", async () => {
    let resolvePromise: any;
    const promise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    (global.fetch as any).mockReturnValueOnce(promise);

    render(<BlockToggle {...defaultProps} />);
    
    const switchElement = screen.getByRole("switch");
    fireEvent.click(switchElement);

    // Switch should be in loading state
    await waitFor(() => {
      expect(switchElement).toHaveClass("ant-switch-loading");
    });

    // Resolve the promise
    resolvePromise({
      ok: true,
      json: async () => ({ blocked: true }),
    });

    await waitFor(() => {
      expect(switchElement).not.toHaveClass("ant-switch-loading");
    });
  });
});

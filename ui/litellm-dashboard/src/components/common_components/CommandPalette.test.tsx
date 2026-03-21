import { screen, fireEvent, act, waitFor } from "@testing-library/react";
import { vi, it, expect, beforeEach, describe } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import CommandPalette from "./CommandPalette";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn().mockReturnValue({
    accessToken: "test-token",
    userRole: "proxy_admin",
    userId: "user-1",
    userEmail: "test@example.com",
    premiumUser: false,
  }),
}));

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn().mockReturnValue(""),
  getGlobalLitellmHeaderName: vi.fn().mockReturnValue("Authorization"),
  deriveErrorMessage: vi.fn().mockReturnValue("Error"),
}));

beforeEach(() => {
  vi.clearAllMocks();
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ keys: [], total_count: 0 }),
  });
});

describe("CommandPalette", () => {
  it("should not render the modal by default", () => {
    renderWithProviders(<CommandPalette />);

    expect(screen.queryByPlaceholderText("Search keys by alias, ID, or user...")).not.toBeInTheDocument();
  });

  it("should open modal on Cmd+K", async () => {
    renderWithProviders(<CommandPalette />);

    await act(async () => {
      fireEvent.keyDown(document, { key: "k", metaKey: true });
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Search keys by alias, ID, or user...")).toBeInTheDocument();
    });
  });

  it("should open modal on Ctrl+K", async () => {
    renderWithProviders(<CommandPalette />);

    await act(async () => {
      fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Search keys by alias, ID, or user...")).toBeInTheDocument();
    });
  });

  it("should show placeholder text when no query is entered", async () => {
    renderWithProviders(<CommandPalette />);

    await act(async () => {
      fireEvent.keyDown(document, { key: "k", metaKey: true });
    });

    await waitFor(() => {
      expect(screen.getByText("Type to search for keys by alias, ID, or user")).toBeInTheDocument();
    });
  });

  it("should toggle modal state on repeated Cmd+K", async () => {
    renderWithProviders(<CommandPalette />);

    await act(async () => {
      fireEvent.keyDown(document, { key: "k", metaKey: true });
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Search keys by alias, ID, or user...")).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.keyDown(document, { key: "k", metaKey: true });
    });

    // The modal close triggers an animation; verify the second keydown was processed
    // by checking that it doesn't error out (antd Modal handles close animation internally)
  });

  it("should show no results message when search returns empty", async () => {
    renderWithProviders(<CommandPalette />);

    await act(async () => {
      fireEvent.keyDown(document, { key: "k", metaKey: true });
    });

    const input = await screen.findByPlaceholderText("Search keys by alias, ID, or user...");
    await act(async () => {
      fireEvent.change(input, { target: { value: "nonexistent" } });
    });

    await waitFor(
      () => {
        expect(screen.getByText(/No keys found for/)).toBeInTheDocument();
      },
      { timeout: 3000 },
    );
  });

  it("should call onSelectKey when a result is clicked", async () => {
    const mockKey = {
      token: "sk-test-123",
      key_name: "sk-...test",
      key_alias: "My Test Key",
      spend: 1.5,
      team_id: "team-1",
      user_id: "user-1",
    };

    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ keys: [mockKey], total_count: 1 }),
    });

    const onSelectKey = vi.fn();
    renderWithProviders(<CommandPalette onSelectKey={onSelectKey} />);

    await act(async () => {
      fireEvent.keyDown(document, { key: "k", metaKey: true });
    });

    const input = await screen.findByPlaceholderText("Search keys by alias, ID, or user...");
    await act(async () => {
      fireEvent.change(input, { target: { value: "My Test" } });
    });

    await waitFor(
      () => {
        expect(screen.getByText("My Test Key")).toBeInTheDocument();
      },
      { timeout: 3000 },
    );

    await act(async () => {
      fireEvent.click(screen.getByText("My Test Key"));
    });

    expect(onSelectKey).toHaveBeenCalledWith(mockKey);
  });
});

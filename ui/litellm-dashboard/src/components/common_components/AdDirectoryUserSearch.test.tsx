import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { AdDirectoryUserSearch } from "./AdDirectoryUserSearch";
import * as networking from "../networking";

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof networking>();
  return {
    ...actual,
    directoryUsersSearchCall: vi.fn(),
  };
});

const mockDirectoryUsersSearchCall = vi.mocked(networking.directoryUsersSearchCall);

describe("AdDirectoryUserSearch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDirectoryUsersSearchCall.mockResolvedValue([]);
  });

  it("should render a single search combobox", () => {
    render(<AdDirectoryUserSearch accessToken="token" />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search by name or email/i)).toBeInTheDocument();
  });

  it("should forward the id prop to the underlying input so a Form label can target it", () => {
    render(<AdDirectoryUserSearch accessToken="token" id="user_email" />);
    expect(screen.getByRole("combobox")).toHaveAttribute("id", "user_email");
  });

  it("should not call directoryUsersSearchCall for a single-character query", async () => {
    const user = userEvent.setup();
    render(<AdDirectoryUserSearch accessToken="token" />);

    await user.type(screen.getByRole("combobox"), "a");

    await waitFor(() => {
      expect(screen.getByText(/no users found/i)).toBeInTheDocument();
    });
    expect(mockDirectoryUsersSearchCall).not.toHaveBeenCalled();
  });

  it("should collapse rapid keystrokes into a single debounced search call", async () => {
    const user = userEvent.setup({ delay: 1 });
    render(<AdDirectoryUserSearch accessToken="token" />);

    await user.type(screen.getByRole("combobox"), "ali");

    await waitFor(() => {
      expect(mockDirectoryUsersSearchCall).toHaveBeenCalledTimes(1);
    });
    expect(mockDirectoryUsersSearchCall).toHaveBeenCalledWith("token", "ali");
  });

  it("should call onChange with the selected user's email", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    mockDirectoryUsersSearchCall.mockResolvedValue([
      { id: "aad-user-id", display_name: "Alice Example", email: "alice@example.com" },
    ]);

    render(<AdDirectoryUserSearch accessToken="token" onChange={onChange} />);

    await user.type(screen.getByRole("combobox"), "ali");
    await user.click(await screen.findByText("Alice Example"));

    expect(onChange).toHaveBeenCalledWith("alice@example.com");
  });

  it("should show the controlled value in the input", () => {
    render(<AdDirectoryUserSearch accessToken="token" value="bob@example.com" />);
    expect(screen.getByRole("combobox")).toHaveValue("bob@example.com");
  });

  it("should show an error message instead of 'No users found' when the search call fails", async () => {
    const user = userEvent.setup();
    mockDirectoryUsersSearchCall.mockRejectedValue(new Error("Microsoft directory search is not configured."));

    render(<AdDirectoryUserSearch accessToken="token" />);

    await user.type(screen.getByRole("combobox"), "ali");

    await waitFor(() => {
      expect(
        screen.getByText(/directory search failed: microsoft directory search is not configured/i),
      ).toBeInTheDocument();
    });
  });

  it("should discard a stale search response that resolves after a newer one", async () => {
    const user = userEvent.setup();
    let resolveFirstSearch: (users: any[]) => void = () => {};
    const firstSearch = new Promise<any[]>((resolve) => {
      resolveFirstSearch = resolve;
    });
    mockDirectoryUsersSearchCall.mockImplementationOnce(() => firstSearch).mockResolvedValueOnce([
      { id: "bob-id", display_name: "Bob Example", email: "bob@example.com" },
    ]);

    render(<AdDirectoryUserSearch accessToken="token" />);

    const searchBox = screen.getByRole("combobox");
    await user.type(searchBox, "ali");
    await waitFor(() => {
      expect(mockDirectoryUsersSearchCall).toHaveBeenCalledWith("token", "ali");
    });

    await user.clear(searchBox);
    await user.type(searchBox, "bob");
    await waitFor(() => {
      expect(mockDirectoryUsersSearchCall).toHaveBeenCalledWith("token", "bob");
    });
    await screen.findByText("Bob Example");

    // The stale "ali" response resolves after the newer "bob" one - it must not overwrite the results.
    resolveFirstSearch([{ id: "alice-id", display_name: "Alice Example", email: "alice@example.com" }]);

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.getByText("Bob Example")).toBeInTheDocument();
    expect(screen.queryByText("Alice Example")).not.toBeInTheDocument();
  });

  it("should clear search results when remounted via a changed key", async () => {
    const user = userEvent.setup();
    mockDirectoryUsersSearchCall.mockResolvedValue([
      { id: "aad-user-id", display_name: "Alice Example", email: "alice@example.com" },
    ]);

    const { rerender } = render(<AdDirectoryUserSearch key={0} accessToken="token" />);

    await user.type(screen.getByRole("combobox"), "ali");
    await user.click(await screen.findByText("Alice Example"));

    rerender(<AdDirectoryUserSearch key={1} accessToken="token" />);

    expect(screen.queryByText("Alice Example")).not.toBeInTheDocument();
  });
});

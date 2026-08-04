import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { DirectoryUserSearch } from "./DirectoryUserSearch";
import * as networking from "../networking";

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof networking>();
  return {
    ...actual,
    directoryUsersSearchCall: vi.fn(),
  };
});

const mockDirectoryUsersSearchCall = vi.mocked(networking.directoryUsersSearchCall);

describe("DirectoryUserSearch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDirectoryUsersSearchCall.mockResolvedValue([]);
  });

  it("should render a single search combobox", () => {
    render(<DirectoryUserSearch accessToken="token" />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search by name or email/i)).toBeInTheDocument();
  });

  it("should forward the id prop to the underlying input so a Form label can target it", () => {
    render(<DirectoryUserSearch accessToken="token" id="user_email" />);
    expect(screen.getByRole("combobox")).toHaveAttribute("id", "user_email");
  });

  it("should not call directoryUsersSearchCall for a single-character query", async () => {
    const user = userEvent.setup();
    render(<DirectoryUserSearch accessToken="token" />);

    await user.type(screen.getByRole("combobox"), "a");

    await waitFor(() => {
      expect(screen.getByText(/no users found/i)).toBeInTheDocument();
    });
    expect(mockDirectoryUsersSearchCall).not.toHaveBeenCalled();
  });

  it("should collapse rapid keystrokes into a single debounced search call", async () => {
    const user = userEvent.setup({ delay: 1 });
    render(<DirectoryUserSearch accessToken="token" />);

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

    render(<DirectoryUserSearch accessToken="token" onChange={onChange} />);

    await user.type(screen.getByRole("combobox"), "ali");
    await user.click(await screen.findByText("Alice Example"));

    expect(onChange).toHaveBeenCalledWith("alice@example.com");
  });

  it("should display the raw email, not the display name, after selecting a result", async () => {
    const user = userEvent.setup();
    mockDirectoryUsersSearchCall.mockResolvedValue([
      { id: "aad-user-id", display_name: "Alice Example", email: "alice@example.com" },
    ]);

    function Controlled() {
      const [value, setValue] = useState("");
      return <DirectoryUserSearch accessToken="token" value={value} onChange={setValue} />;
    }
    render(<Controlled />);

    await user.type(screen.getByRole("combobox"), "ali");
    await user.click(await screen.findByText("Alice Example"));

    expect(screen.getByRole("combobox")).toHaveValue("alice@example.com");
  });

  it("should show the controlled value in the input", () => {
    render(<DirectoryUserSearch accessToken="token" value="bob@example.com" />);
    expect(screen.getByRole("combobox")).toHaveValue("bob@example.com");
  });

  it("should show an error message instead of 'No users found' when the search call fails", async () => {
    const user = userEvent.setup();
    mockDirectoryUsersSearchCall.mockRejectedValue(new Error("Directory search is not configured."));

    render(<DirectoryUserSearch accessToken="token" />);

    await user.type(screen.getByRole("combobox"), "ali");

    await waitFor(() => {
      expect(screen.getByText(/directory search failed: directory search is not configured/i)).toBeInTheDocument();
    });
  });

  it("should discard a stale search response that resolves after a newer one", async () => {
    const user = userEvent.setup();
    let resolveFirstSearch: (users: any[]) => void = () => {};
    const firstSearch = new Promise<any[]>((resolve) => {
      resolveFirstSearch = resolve;
    });
    mockDirectoryUsersSearchCall
      .mockImplementationOnce(() => firstSearch)
      .mockResolvedValueOnce([{ id: "bob-id", display_name: "Bob Example", email: "bob@example.com" }]);

    render(<DirectoryUserSearch accessToken="token" />);

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

    const { rerender } = render(<DirectoryUserSearch key={0} accessToken="token" />);

    await user.type(screen.getByRole("combobox"), "ali");
    await user.click(await screen.findByText("Alice Example"));

    rerender(<DirectoryUserSearch key={1} accessToken="token" />);

    expect(screen.queryByText("Alice Example")).not.toBeInTheDocument();
  });
});

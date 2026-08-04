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

  it("should render the search combobox", () => {
    render(<AdDirectoryUserSearch accessToken="token" resetSignal={0} onSelectUser={vi.fn()} />);
    expect(screen.getByRole("combobox", { name: /user email/i })).toBeInTheDocument();
  });

  it("should not call directoryUsersSearchCall for a single-character query", async () => {
    const user = userEvent.setup();
    render(<AdDirectoryUserSearch accessToken="token" resetSignal={0} onSelectUser={vi.fn()} />);

    await user.type(screen.getByRole("combobox", { name: /user email/i }), "a");

    await waitFor(() => {
      expect(screen.getByText(/no users found/i)).toBeInTheDocument();
    });
    expect(mockDirectoryUsersSearchCall).not.toHaveBeenCalled();
  });

  it("should collapse rapid keystrokes into a single debounced search call", async () => {
    const user = userEvent.setup({ delay: 1 });
    render(<AdDirectoryUserSearch accessToken="token" resetSignal={0} onSelectUser={vi.fn()} />);

    await user.type(screen.getByRole("combobox", { name: /user email/i }), "ali");

    await waitFor(() => {
      expect(mockDirectoryUsersSearchCall).toHaveBeenCalledTimes(1);
    });
    expect(mockDirectoryUsersSearchCall).toHaveBeenCalledWith("token", "ali");
  });

  it("should call onSelectUser with the chosen directory user", async () => {
    const user = userEvent.setup();
    const onSelectUser = vi.fn();
    mockDirectoryUsersSearchCall.mockResolvedValue([
      { id: "aad-user-id", display_name: "Alice Example", email: "alice@example.com" },
    ]);

    render(<AdDirectoryUserSearch accessToken="token" resetSignal={0} onSelectUser={onSelectUser} />);

    await user.type(screen.getByRole("combobox", { name: /user email/i }), "ali");
    await user.click(await screen.findByText("Alice Example"));

    expect(onSelectUser).toHaveBeenCalledWith({
      id: "aad-user-id",
      display_name: "Alice Example",
      email: "alice@example.com",
    });
  });

  it("should show an error message instead of 'No users found' when the search call fails", async () => {
    const user = userEvent.setup();
    mockDirectoryUsersSearchCall.mockRejectedValue(new Error("Microsoft directory search is not configured."));

    render(<AdDirectoryUserSearch accessToken="token" resetSignal={0} onSelectUser={vi.fn()} />);

    await user.type(screen.getByRole("combobox", { name: /user email/i }), "ali");

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

    render(<AdDirectoryUserSearch accessToken="token" resetSignal={0} onSelectUser={vi.fn()} />);

    const searchBox = screen.getByRole("combobox", { name: /user email/i });
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

  it("should clear the selected value and options when resetSignal changes", async () => {
    const user = userEvent.setup();
    mockDirectoryUsersSearchCall.mockResolvedValue([
      { id: "aad-user-id", display_name: "Alice Example", email: "alice@example.com" },
    ]);

    const { rerender } = render(
      <AdDirectoryUserSearch accessToken="token" resetSignal={0} onSelectUser={vi.fn()} />,
    );

    await user.type(screen.getByRole("combobox", { name: /user email/i }), "ali");
    await user.click(await screen.findByText("Alice Example"));

    rerender(<AdDirectoryUserSearch accessToken="token" resetSignal={1} onSelectUser={vi.fn()} />);

    expect(screen.queryByText("Alice Example")).not.toBeInTheDocument();
  });
});

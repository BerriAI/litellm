import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import UserSingleSelect from "./user_single_select";

const mockUseInfiniteUsers = vi.fn();

vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useInfiniteUsers: (...args: any[]) => mockUseInfiniteUsers(...args),
}));

const buildPage = (users: Array<{ user_id: string; user_email: string | null; user_alias: string | null }>) => ({
  users,
  page: 1,
  page_size: 50,
  total: users.length,
  total_pages: 1,
});

const DEFAULT_HOOK_STATE = {
  data: {
    pages: [
      buildPage([
        { user_id: "user-1", user_email: "alice@example.com", user_alias: null },
        { user_id: "user-2", user_email: null, user_alias: "Bob" },
        { user_id: "user-3", user_email: "charlie@example.com", user_alias: "Charlie" },
      ]),
    ],
  },
  fetchNextPage: vi.fn(),
  hasNextPage: false,
  isFetchingNextPage: false,
  isLoading: false,
};

describe("UserSingleSelect", () => {
  it("should render a combobox", () => {
    mockUseInfiniteUsers.mockReturnValue(DEFAULT_HOOK_STATE);
    render(<UserSingleSelect />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should display user options when opened", async () => {
    mockUseInfiniteUsers.mockReturnValue(DEFAULT_HOOK_STATE);
    const user = userEvent.setup();
    render(<UserSingleSelect />);

    await act(async () => {
      await user.click(screen.getByRole("combobox"));
    });

    expect(await screen.findByText("alice@example.com")).toBeInTheDocument();
    expect(screen.getByText("user-1")).toBeInTheDocument();
  });

  it("should call onChange with user_id when a user is selected", async () => {
    mockUseInfiniteUsers.mockReturnValue(DEFAULT_HOOK_STATE);
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<UserSingleSelect onChange={onChange} />);

    await act(async () => {
      await user.click(screen.getByRole("combobox"));
    });

    await act(async () => {
      await user.click(await screen.findByText("alice@example.com"));
    });

    expect(onChange).toHaveBeenCalledWith("user-1");
  });

  it("should call onChange with null when selection is cleared", async () => {
    mockUseInfiniteUsers.mockReturnValue(DEFAULT_HOOK_STATE);
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<UserSingleSelect value="user-1" onChange={onChange} />);

    const clearButton = document.querySelector(".ant-select-clear");
    if (clearButton) {
      await act(async () => {
        await user.click(clearButton as Element);
      });
      expect(onChange).toHaveBeenCalledWith(null);
    }
  });

  it("should pass search input to useInfiniteUsers as debounced value", async () => {
    mockUseInfiniteUsers.mockReturnValue({ ...DEFAULT_HOOK_STATE, data: { pages: [] } });
    const user = userEvent.setup();
    render(<UserSingleSelect />);

    await act(async () => {
      await user.click(screen.getByRole("combobox"));
      await user.type(screen.getByRole("combobox"), "alice");
    });

    await waitFor(() => {
      expect(mockUseInfiniteUsers).toHaveBeenCalledWith(
        expect.any(Number),
        "alice",
      );
    });
  });

  it("should show loading indicator when isLoading is true", () => {
    mockUseInfiniteUsers.mockReturnValue({ ...DEFAULT_HOOK_STATE, isLoading: true });
    render(<UserSingleSelect />);
    expect(document.querySelector(".ant-select-loading")).toBeInTheDocument();
  });

  it("should show user alias in label when alias is set", async () => {
    mockUseInfiniteUsers.mockReturnValue(DEFAULT_HOOK_STATE);
    const user = userEvent.setup();
    render(<UserSingleSelect />);

    await act(async () => {
      await user.click(screen.getByRole("combobox"));
    });

    expect(await screen.findByText("charlie@example.com")).toBeInTheDocument();
  });

  it("should use custom placeholder when provided", () => {
    mockUseInfiniteUsers.mockReturnValue(DEFAULT_HOOK_STATE);
    render(<UserSingleSelect placeholder="Find a user..." />);
    expect(screen.getByText("Find a user...")).toBeInTheDocument();
  });

  it("should add ant-select-disabled class when disabled prop is true", () => {
    mockUseInfiniteUsers.mockReturnValue(DEFAULT_HOOK_STATE);
    const { container } = render(<UserSingleSelect disabled />);
    expect(container.querySelector(".ant-select-disabled")).toBeTruthy();
  });

  it("should show fetchNextPage spinner when isFetchingNextPage is true", async () => {
    mockUseInfiniteUsers.mockReturnValue({
      ...DEFAULT_HOOK_STATE,
      hasNextPage: true,
      isFetchingNextPage: true,
    });
    const user = userEvent.setup();
    render(<UserSingleSelect />);

    await act(async () => {
      await user.click(screen.getByRole("combobox"));
    });

    const spinners = document.querySelectorAll(".anticon-loading");
    expect(spinners.length).toBeGreaterThanOrEqual(1);
  });
});

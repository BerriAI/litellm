import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useInfiniteUsers, useUserLookup } from "@/app/(dashboard)/hooks/users/useUsers";
import type { UserInfo } from "@/components/networking";
import UserDropdown, { userOptionLabel } from "./UserDropdown";

vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useInfiniteUsers: vi.fn(),
  useUserLookup: vi.fn(),
}));

const userRow = (userId: string, overrides: Partial<Pick<UserInfo, "user_alias" | "user_email">> = {}) => ({
  user_id: userId,
  user_alias: null,
  user_email: "",
  ...overrides,
});

const ALIASED = userRow("user-1", { user_alias: "Alice Admin", user_email: "alice@example.com" });
const EMAILED = userRow("user-2", { user_email: "bob@example.com" });
const BARE = userRow("user-3");

const mockUsersResult = (
  overrides: Partial<{
    pages: { users: ReturnType<typeof userRow>[] }[];
    fetchNextPage: () => void;
    isLoading: boolean;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
  }> = {},
) => {
  const { pages = [{ users: [ALIASED, EMAILED, BARE] }], ...rest } = overrides;
  return {
    data: { pages },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
    ...rest,
  } as unknown as ReturnType<typeof useInfiniteUsers>;
};

const mockLookupResult = (user: ReturnType<typeof userRow> | null) =>
  ({ data: user }) as unknown as ReturnType<typeof useUserLookup>;

function setListMetrics(list: HTMLElement, metrics: { scrollTop: number; clientHeight: number; scrollHeight: number }) {
  Object.defineProperty(list, "scrollTop", { value: metrics.scrollTop, configurable: true });
  Object.defineProperty(list, "clientHeight", { value: metrics.clientHeight, configurable: true });
  Object.defineProperty(list, "scrollHeight", { value: metrics.scrollHeight, configurable: true });
}

describe("UserDropdown", () => {
  const mockUseInfiniteUsers = vi.mocked(useInfiniteUsers);
  const mockUseUserLookup = vi.mocked(useUserLookup);

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseInfiniteUsers.mockReturnValue(mockUsersResult());
    mockUseUserLookup.mockReturnValue(mockLookupResult(null));
  });

  const combobox = () => screen.getByRole("combobox");

  it("queries the first page of users with no search term", () => {
    render(<UserDropdown onChange={vi.fn()} />);

    expect(mockUseInfiniteUsers).toHaveBeenCalledWith(50, undefined);
  });

  it("forwards the pageSize prop to the users query", () => {
    render(<UserDropdown onChange={vi.fn()} pageSize={25} />);

    expect(mockUseInfiniteUsers).toHaveBeenCalledWith(25, undefined);
  });

  it("sends the typed query to the server instead of narrowing the loaded page", async () => {
    const user = userEvent.setup();
    render(<UserDropdown onChange={vi.fn()} />);

    await user.click(combobox());
    fireEvent.change(combobox(), { target: { value: "alice" } });

    await waitFor(() => expect(mockUseInfiniteUsers).toHaveBeenCalledWith(50, "alice"));
    expect(screen.getByText("bob@example.com (user-2)")).toBeInTheDocument();
  });

  it("labels a user by alias, falling back to email and then the bare id", async () => {
    const user = userEvent.setup();
    render(<UserDropdown onChange={vi.fn()} />);

    await user.click(combobox());

    expect(screen.getByText("Alice Admin (user-1)")).toBeInTheDocument();
    expect(screen.getByText("bob@example.com (user-2)")).toBeInTheDocument();
    expect(screen.getByText("user-3")).toBeInTheDocument();
  });

  it("deduplicates a user that appears on more than one page", async () => {
    mockUseInfiniteUsers.mockReturnValue(
      mockUsersResult({
        pages: [{ users: [ALIASED] }, { users: [ALIASED, EMAILED] }],
      }),
    );
    const user = userEvent.setup();
    render(<UserDropdown onChange={vi.fn()} />);

    await user.click(combobox());

    expect(screen.getAllByText("Alice Admin (user-1)")).toHaveLength(1);
    expect(screen.getByText("bob@example.com (user-2)")).toBeInTheDocument();
  });

  it("loads the next page once the list is scrolled near the bottom", async () => {
    const fetchNextPage = vi.fn();
    mockUseInfiniteUsers.mockReturnValue(mockUsersResult({ fetchNextPage, hasNextPage: true }));
    const user = userEvent.setup();
    render(<UserDropdown onChange={vi.fn()} />);

    await user.click(combobox());
    const list = await screen.findByTestId("paginated-search-select-list");

    setListMetrics(list, { scrollTop: 0, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(fetchNextPage).not.toHaveBeenCalled();

    setListMetrics(list, { scrollTop: 850, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(fetchNextPage).toHaveBeenCalledTimes(1);
  });

  it("does not load more when there is no next page or one is already in flight", async () => {
    const fetchNextPage = vi.fn();
    mockUseInfiniteUsers.mockReturnValue(mockUsersResult({ fetchNextPage, hasNextPage: false }));
    const user = userEvent.setup();
    const { unmount } = render(<UserDropdown onChange={vi.fn()} />);

    await user.click(combobox());
    setListMetrics(await screen.findByTestId("paginated-search-select-list"), {
      scrollTop: 900,
      clientHeight: 100,
      scrollHeight: 1000,
    });
    fireEvent.scroll(screen.getByTestId("paginated-search-select-list"));
    expect(fetchNextPage).not.toHaveBeenCalled();
    unmount();

    mockUseInfiniteUsers.mockReturnValue(
      mockUsersResult({ fetchNextPage, hasNextPage: true, isFetchingNextPage: true }),
    );
    render(<UserDropdown onChange={vi.fn()} />);

    await user.click(combobox());
    setListMetrics(await screen.findByTestId("paginated-search-select-list"), {
      scrollTop: 900,
      clientHeight: 100,
      scrollHeight: 1000,
    });
    fireEvent.scroll(screen.getByTestId("paginated-search-select-list"));
    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it("reports the picked user id to onChange", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<UserDropdown onChange={onChange} />);

    await user.click(combobox());
    await user.click(screen.getByText("bob@example.com (user-2)"));

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("user-2");
  });

  it("reports null to onChange when the selection is cleared", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<UserDropdown onChange={onChange} value="user-1" />);

    await user.click(document.querySelector('[data-slot="combobox-clear"]') as HTMLElement);

    expect(onChange).toHaveBeenCalledWith(null);
  });

  it("shows the label of the user passed in value", () => {
    render(<UserDropdown onChange={vi.fn()} value="user-1" />);

    expect(combobox()).toHaveValue("Alice Admin (user-1)");
    expect(mockUseUserLookup).toHaveBeenLastCalledWith(null);
  });

  it("looks up a selected user that is not on the loaded page so its label still shows", () => {
    const OFF_PAGE = userRow("user-99", { user_alias: "Zed Offpage" });
    mockUseUserLookup.mockReturnValue(mockLookupResult(OFF_PAGE));
    render(<UserDropdown onChange={vi.fn()} value="user-99" />);

    expect(mockUseUserLookup).toHaveBeenLastCalledWith("user-99");
    expect(combobox()).toHaveValue("Zed Offpage (user-99)");
  });

  it("falls back to the bare id while the off-page lookup has not resolved", () => {
    render(<UserDropdown onChange={vi.fn()} value="user-99" />);

    expect(combobox()).toHaveValue("user-99");
  });
});

describe("userOptionLabel", () => {
  it("prefers the alias", () => {
    expect(userOptionLabel(ALIASED)).toBe("Alice Admin (user-1)");
  });

  it("falls back to the email", () => {
    expect(userOptionLabel(EMAILED)).toBe("bob@example.com (user-2)");
  });

  it("falls back to the bare id", () => {
    expect(userOptionLabel(BARE)).toBe("user-3");
  });
});

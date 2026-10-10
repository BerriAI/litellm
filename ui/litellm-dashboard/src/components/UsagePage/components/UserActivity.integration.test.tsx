import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { DailyActivityUserPageResponse, UserActivityRow } from "../dailyActivityApi";
import TopUsersView from "./TopUsersView";
import UserActivityPanel from "./UserActivityPanel";
import type { FetchUserPage } from "./userActivityData";

const userRow = (overrides: Partial<UserActivityRow>): UserActivityRow => ({
  user_id: "user-1",
  user_email: null,
  user_alias: null,
  spend: 1,
  prompt_tokens: 10,
  completion_tokens: 5,
  total_tokens: 15,
  api_requests: 3,
  successful_requests: 3,
  failed_requests: 0,
  ...overrides,
});

const fullPage = (prefix: string, count = 50): UserActivityRow[] =>
  Array.from({ length: count }, (_, index) => userRow({ user_id: `${prefix}-${index}` }));

const pageOf = (users: UserActivityRow[], total?: number): DailyActivityUserPageResponse => ({
  users,
  total_users: total ?? users.length,
  offset: 0,
  limit: 50,
});

describe("TopUsersView", () => {
  it("renders rows in the order the API returns them", async () => {
    const fetchUserPage: FetchUserPage = vi
      .fn()
      .mockResolvedValue(
        pageOf([
          userRow({ user_id: "big-spender", user_email: "big@example.com", spend: 50 }),
          userRow({ user_id: "small-spender", spend: 5 }),
        ]),
      );

    render(<TopUsersView fetchUserPage={fetchUserPage} />);

    expect(await screen.findByText("big@example.com")).toBeInTheDocument();
    const labels = screen.getAllByRole("listitem").map((item) => item.textContent);
    expect(labels[0]).toContain("big@example.com");
    expect(labels[1]).toContain("small-spender");
    expect(fetchUserPage).toHaveBeenCalledWith(0, 10);
  });

  it("shows column headers and renders sub-cent spend as < $0.01 while zero stays $0.00", async () => {
    const fetchUserPage: FetchUserPage = vi
      .fn()
      .mockResolvedValue(pageOf([userRow({ user_id: "tiny", spend: 0.004 }), userRow({ user_id: "zero", spend: 0 })]));

    render(<TopUsersView fetchUserPage={fetchUserPage} />);

    expect(await screen.findByText("tiny")).toBeInTheDocument();
    expect(screen.getByText("User")).toBeInTheDocument();
    expect(screen.getByText("Spend")).toBeInTheDocument();
    expect(screen.getByText("Requests")).toBeInTheDocument();
    expect(screen.getByText("< $0.01")).toBeInTheDocument();
    expect(screen.getByText("$0.00")).toBeInTheDocument();
  });

  it("falls back from email to alias to user_id to a placeholder", async () => {
    const fetchUserPage: FetchUserPage = vi
      .fn()
      .mockResolvedValue(
        pageOf([
          userRow({ user_id: "u-email", user_email: "e@example.com", user_alias: "alias-hidden" }),
          userRow({ user_id: "u-alias", user_alias: "Only Alias" }),
          userRow({ user_id: "u-bare" }),
          userRow({ user_id: null }),
        ]),
      );

    render(<TopUsersView fetchUserPage={fetchUserPage} />);

    expect(await screen.findByText("e@example.com")).toBeInTheDocument();
    expect(screen.getByText("u-email")).toBeInTheDocument();
    expect(screen.getByText("Only Alias")).toBeInTheDocument();
    expect(screen.getByText("u-alias")).toBeInTheDocument();
    expect(screen.getByText("u-bare")).toBeInTheDocument();
    expect(screen.getByText("(no user)")).toBeInTheDocument();
  });

  it("shows an error state with retry", async () => {
    const fetchUserPage: FetchUserPage = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue(pageOf([userRow({ user_id: "u-1" })]));

    render(<TopUsersView fetchUserPage={fetchUserPage} />);

    expect(await screen.findByText(/Could not load top users/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Retry"));
    expect(await screen.findByText("u-1")).toBeInTheDocument();
  });
});

describe("UserActivityPanel", () => {
  it("renders paged rows with the user count and columns", async () => {
    const fetchUserPage: FetchUserPage = vi
      .fn()
      .mockResolvedValue(
        pageOf([userRow({ user_id: "first", spend: 9 }), userRow({ user_id: "second", spend: 1 })], 120),
      );

    render(<UserActivityPanel fetchUserPage={fetchUserPage} />);

    expect(await screen.findByText("first")).toBeInTheDocument();
    expect(screen.getByText("second")).toBeInTheDocument();
    expect(screen.getByText("120 users")).toBeInTheDocument();
    const rows = screen.getAllByRole("listitem").map((item) => item.textContent);
    expect(rows[0]).toContain("first");
    expect(rows[1]).toContain("second");
  });

  it("loads the next page with offset 50 when Load more is clicked", async () => {
    const fetchUserPage: FetchUserPage = vi.fn().mockImplementation((offset: number) => {
      const page = {
        users: offset === 0 ? fullPage("first") : [userRow({ user_id: "later" })],
        total_users: 51,
        offset,
        limit: 50,
      };
      return Promise.resolve(page);
    });

    render(<UserActivityPanel fetchUserPage={fetchUserPage} />);

    fireEvent.click(await screen.findByText("Load more"));
    expect(fetchUserPage).toHaveBeenLastCalledWith(50, 50);
    expect(await screen.findByText("later")).toBeInTheDocument();
    expect(screen.getByText("first-0")).toBeInTheDocument();
  });

  it("deduplicates a user repeated across pages but keeps the raw offset", async () => {
    const fetchUserPage: FetchUserPage = vi.fn().mockImplementation((offset: number) => {
      const page = {
        users: offset === 0 ? fullPage("first") : [userRow({ user_id: "first-7" }), userRow({ user_id: "later" })],
        total_users: 52,
        offset,
        limit: 50,
      };
      return Promise.resolve(page);
    });

    render(<UserActivityPanel fetchUserPage={fetchUserPage} />);

    fireEvent.click(await screen.findByText("Load more"));
    expect(await screen.findByText("later")).toBeInTheDocument();
    const occurrences = screen.getAllByRole("listitem").filter((item) => item.textContent?.includes("first-7"));
    expect(occurrences).toHaveLength(1);
    expect(fetchUserPage).toHaveBeenLastCalledWith(50, 50);
    expect(screen.queryByText("Load more")).not.toBeInTheDocument();
  });

  it("keeps a literal __no_user__ id and a null id as separate rows", async () => {
    const fetchUserPage: FetchUserPage = vi.fn().mockImplementation((offset: number) => {
      const page = {
        users:
          offset === 0
            ? [userRow({ user_id: null }), ...fullPage("first", 49)]
            : [userRow({ user_id: "__no_user__", user_alias: "Named No User" })],
        total_users: 51,
        offset,
        limit: 50,
      };
      return Promise.resolve(page);
    });

    render(<UserActivityPanel fetchUserPage={fetchUserPage} />);

    fireEvent.click(await screen.findByText("Load more"));
    expect(await screen.findByText("Named No User")).toBeInTheDocument();
    expect(screen.getByText("(no user)")).toBeInTheDocument();
    expect(await screen.findByText("__no_user__")).toBeInTheDocument();
  });

  it("restarts at offset 0 when the fetcher changes", async () => {
    const firstFetcher: FetchUserPage = vi.fn().mockResolvedValue(pageOf(fullPage("p1"), 100));
    const secondFetcher: FetchUserPage = vi.fn().mockResolvedValue(pageOf([userRow({ user_id: "refetched" })]));

    const { rerender } = render(<UserActivityPanel fetchUserPage={firstFetcher} />);
    fireEvent.click(await screen.findByText("Load more"));
    await waitFor(() => expect(firstFetcher).toHaveBeenLastCalledWith(50, 50));

    rerender(<UserActivityPanel fetchUserPage={secondFetcher} />);

    await waitFor(() => expect(secondFetcher).toHaveBeenCalledWith(0, 50));
    expect(await screen.findByText("refetched")).toBeInTheDocument();
  });

  it("ignores a stale first page that resolves after the fetcher changed", async () => {
    let releaseStale: (value: DailyActivityUserPageResponse) => void = () => {};
    const staleFetcher: FetchUserPage = vi
      .fn()
      .mockImplementation(() => new Promise<DailyActivityUserPageResponse>((resolve) => (releaseStale = resolve)));
    const freshFetcher: FetchUserPage = vi.fn().mockResolvedValue(pageOf([userRow({ user_id: "fresh" })]));

    const { rerender } = render(<UserActivityPanel fetchUserPage={staleFetcher} />);
    rerender(<UserActivityPanel fetchUserPage={freshFetcher} />);
    await waitFor(() => expect(freshFetcher).toHaveBeenCalled());
    releaseStale(pageOf([userRow({ user_id: "stale" })]));

    expect(await screen.findByText("fresh")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText("stale")).not.toBeInTheDocument());
  });

  it("shows an error state with retry on first-page failure", async () => {
    const fetchUserPage: FetchUserPage = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue(pageOf([userRow({ user_id: "recovered" })]));

    render(<UserActivityPanel fetchUserPage={fetchUserPage} />);

    expect(await screen.findByText(/Could not load users for this range/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Retry"));
    expect(await screen.findByText("recovered")).toBeInTheDocument();
  });
});

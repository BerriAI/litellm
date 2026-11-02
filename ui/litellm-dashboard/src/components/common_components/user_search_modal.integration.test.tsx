import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders as render } from "../../../tests/test-utils";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import UserSearchModal from "./user_search_modal";
import { userFilterUICall } from "@/components/networking";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

vi.mock("@/components/networking", () => ({
  userFilterUICall: vi.fn().mockResolvedValue([]),
}));

const renderModal = () =>
  render(<UserSearchModal isVisible onCancel={vi.fn()} onSubmit={vi.fn()} accessToken="sk-test" />);

const getEmailSearchInput = () => within(screen.getByTestId("member-email-search")).getByRole("combobox");

describe("UserSearchModal", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(userFilterUICall).mockClear();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  it("debounces the user search and fires exactly once with the last typed value", async () => {
    renderModal();
    const input = getEmailSearchInput();

    act(() => {
      fireEvent.change(input, { target: { value: "a" } });
      fireEvent.change(input, { target: { value: "ab" } });
      fireEvent.change(input, { target: { value: "abc" } });
    });

    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_WAIT_MS - 1);
    });
    expect(userFilterUICall).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(1);
      await Promise.resolve();
    });

    expect(userFilterUICall).toHaveBeenCalledTimes(1);
    const params = vi.mocked(userFilterUICall).mock.calls[0][1];
    expect(params.get("user_email")).toBe("abc");
  });

  it("does not fire the search when unmounted mid-wait", () => {
    const { unmount } = renderModal();
    const input = getEmailSearchInput();

    act(() => {
      fireEvent.change(input, { target: { value: "abc" } });
    });

    unmount();

    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_WAIT_MS * 2);
    });

    expect(userFilterUICall).not.toHaveBeenCalled();
  });

  it("explains scoped search and exact-email enrollment", () => {
    renderModal();

    const notice = screen.getByRole("alert");
    expect(notice).toHaveTextContent(/search for users you can access/i);
    expect(notice).toHaveTextContent(/enter a complete email address/i);
    // info, not warning: a warning here would read as an error state on an empty form
    expect(notice).toHaveAttribute("data-variant", "info");
  });
});

describe("UserSearchModal submit payload", () => {
  beforeEach(() => {
    vi.mocked(userFilterUICall).mockReset();
    vi.mocked(userFilterUICall).mockResolvedValue([{ user_id: "u-1", user_email: "picked@example.com" }] as never);
  });

  const setup = () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<UserSearchModal isVisible onCancel={vi.fn()} onSubmit={onSubmit} accessToken="sk-test" />);
    return { user, onSubmit };
  };

  const save = () => screen.getByRole("button", { name: /add member/i });

  const searchByEmail = async (user: ReturnType<typeof userEvent.setup>, text: string) => {
    const input = getEmailSearchInput();
    await user.click(input);
    await user.type(input, text);
    await waitFor(() => expect(userFilterUICall).toHaveBeenCalled(), { timeout: 3000 });
    await user.click(await screen.findByRole("option", { name: "picked@example.com" }));
  };

  it("should block submission without a selected user and after clearing the paired identity", async () => {
    const { user, onSubmit } = setup();

    expect(save()).toBeDisabled();
    await searchByEmail(user, "pick");
    await user.click(screen.getAllByRole("button", { name: "Clear" })[0]);
    expect(screen.getByLabelText("Email")).toHaveValue("");
    expect(screen.getByLabelText("User ID")).toHaveValue("");
    expect(save()).toBeDisabled();
    await user.click(save());

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it.each(["empty", "forbidden"])("submits an exact email when directory search is %s", async (result) => {
    vi.mocked(userFilterUICall).mockReset();
    if (result === "forbidden") {
      vi.mocked(userFilterUICall).mockRejectedValue(new Error("403 Forbidden"));
    } else {
      vi.mocked(userFilterUICall).mockResolvedValue([]);
    }
    const { user, onSubmit } = setup();
    const input = getEmailSearchInput();
    await user.click(input);
    await user.type(input, "invite@example.com");
    await user.click(await screen.findByRole("option", { name: "Use email invite@example.com" }));
    await user.click(save());

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledExactlyOnceWith({
        user_email: "invite@example.com",
        user_id: null,
        role: "user",
      }),
    );
  });

  it("clears the previous user ID when selecting an email outside search results", async () => {
    const { user, onSubmit } = setup();
    await searchByEmail(user, "pick");
    vi.mocked(userFilterUICall).mockResolvedValue([]);
    await user.click(getEmailSearchInput());
    await user.type(getEmailSearchInput(), "invite@example.com");
    await user.click(await screen.findByRole("option", { name: "Use email invite@example.com" }));
    await user.click(save());

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledExactlyOnceWith({
        user_email: "invite@example.com",
        user_id: null,
        role: "user",
      }),
    );
  });

  it.each(["partial", "invalid@"])("does not offer an invitation for %s", async (email) => {
    vi.mocked(userFilterUICall).mockResolvedValue([]);
    const { user } = setup();
    await user.click(getEmailSearchInput());
    await user.type(getEmailSearchInput(), email);
    await screen.findByText("No results");

    expect(screen.queryByRole("option")).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(save()).toBeDisabled();
  });

  it("carries the picked user's email and id into the payload", async () => {
    const { user, onSubmit } = setup();

    await searchByEmail(user, "pick");
    await user.click(save());

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toStrictEqual({
      user_email: "picked@example.com",
      user_id: "u-1",
      role: "user",
    });
  });

  it("preserves a missing email when selecting an existing user by ID", async () => {
    vi.mocked(userFilterUICall).mockResolvedValue([{ user_id: "id-only-user", user_email: null }]);
    const { user, onSubmit } = setup();
    const input = screen.getByLabelText("User ID");
    await user.click(input);
    await user.type(input, "id-only");
    await user.click(await screen.findByRole("option", { name: "id-only-user" }));
    await user.click(save());

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledExactlyOnceWith({
        user_email: null,
        user_id: "id-only-user",
        role: "user",
      }),
    );
  });

  it("carries a role changed off its default into the payload", async () => {
    const { onSubmit } = setup();
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

    await searchByEmail(user, "pick");
    await user.click(screen.getByLabelText("Member Role"));
    await user.click(await screen.findByRole("option", { name: /^admin/ }));
    await user.click(save());

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toMatchObject({ role: "admin" });
  });

  it("keeps the picked identity when the option showing the current value is reselected", async () => {
    const { user, onSubmit } = setup();

    await searchByEmail(user, "pick");

    vi.mocked(userFilterUICall).mockResolvedValue([] as never);
    const idInput = screen.getByLabelText("User ID");
    await user.click(idInput);
    await user.type(idInput, "zzz");
    await waitFor(() => expect(userFilterUICall).toHaveBeenCalledTimes(2), { timeout: 3000 });

    await user.click(screen.getByPlaceholderText("Search by email"));
    await user.click(await screen.findByRole("option", { name: "picked@example.com" }));

    await user.click(save());

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      user_email: "picked@example.com",
      user_id: "u-1",
    });
  });

  it("commits the first match when the typed search is confirmed with Enter", async () => {
    const { user, onSubmit } = setup();

    const input = getEmailSearchInput();
    await user.click(input);
    await user.type(input, "pick");
    await waitFor(() => expect(userFilterUICall).toHaveBeenCalled(), { timeout: 3000 });
    await screen.findByRole("option", { name: "picked@example.com" });

    await user.keyboard("{Enter}");
    await user.click(save());

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toStrictEqual({
      user_email: "picked@example.com",
      user_id: "u-1",
      role: "user",
    });
  });

  it("does not submit on Enter in any field, while the button still does", async () => {
    const { user, onSubmit } = setup();

    await searchByEmail(user, "pick");
    await user.click(getEmailSearchInput());
    await user.keyboard("{Enter}");
    await user.click(screen.getByLabelText("User ID"));
    await user.keyboard("{Enter}");
    await user.click(screen.getByLabelText("Member Role"));
    await user.keyboard("{Escape}");
    await user.keyboard("{Enter}");
    expect(onSubmit).not.toHaveBeenCalled();

    await user.click(save());

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
  });
});

describe("UserSearchModal search lifecycle", () => {
  const directory = [
    { user_id: "u-jones", user_email: "alice.jones@example.com" },
    { user_id: "u-smith", user_email: "alice.smith@example.com" },
    { user_id: "u-bob", user_email: "bob@example.com" },
  ];

  beforeEach(() => {
    vi.mocked(userFilterUICall).mockReset();
    vi.mocked(userFilterUICall).mockImplementation((_accessToken, params) => {
      const query = params.get("user_email") ?? "";
      return Promise.resolve(directory.filter((user) => user.user_email.includes(query))) as never;
    });
  });

  const searchedFor = (): string[] =>
    vi.mocked(userFilterUICall).mock.calls.map((call) => {
      const email = call[1].get("user_email");
      return email === null ? `user_id=${call[1].get("user_id")}` : `user_email=${email}`;
    });

  const settleDebounce = () =>
    act(async () => {
      await new Promise((resolve) => setTimeout(resolve, DEBOUNCE_WAIT_MS + 100));
    });

  it("leaves the search unfiltered after a pick, so reopening searches the newly typed text", async () => {
    const user = userEvent.setup();
    render(<UserSearchModal isVisible onCancel={vi.fn()} onSubmit={vi.fn()} accessToken="sk-test" />);

    const input = getEmailSearchInput();
    await user.click(input);
    await user.type(input, "ali");
    await user.click(await screen.findByRole("option", { name: "alice.jones@example.com" }));

    await settleDebounce();
    expect(searchedFor()).toEqual(["user_email=ali"]);

    await user.click(input);
    await user.type(input, "bob");

    expect(await screen.findByRole("option", { name: "bob@example.com" })).toBeInTheDocument();
    expect(searchedFor()).toEqual(["user_email=ali", "user_email=bob"]);
  });
});

describe("UserSearchModal out-of-order search results", () => {
  const answers = new Map<string, (users: { user_id: string; user_email: string }[]) => void>();

  beforeEach(() => {
    answers.clear();
    vi.mocked(userFilterUICall).mockReset();
    vi.mocked(userFilterUICall).mockImplementation(
      (_accessToken, params) =>
        new Promise((resolve) => {
          answers.set(params.get("user_email") ?? "", resolve);
        }) as never,
    );
  });

  const answerFor = async (search: string, users: { user_id: string; user_email: string }[]) => {
    const resolve = answers.get(search);
    if (resolve === undefined) throw new Error(`no pending search for "${search}"`);
    await act(async () => {
      resolve(users);
    });
  };

  it("commits the current search's match when an abandoned search answers last", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<UserSearchModal isVisible onCancel={vi.fn()} onSubmit={onSubmit} accessToken="sk-test" />);

    const input = getEmailSearchInput();
    await user.click(input);
    await user.type(input, "ali");
    await waitFor(() => expect(answers.has("ali")).toBe(true), { timeout: 3000 });

    await user.type(input, "ce.smith@example.com");
    await waitFor(() => expect(answers.has("alice.smith@example.com")).toBe(true), { timeout: 3000 });

    await answerFor("alice.smith@example.com", [{ user_id: "u-smith", user_email: "alice.smith@example.com" }]);
    await screen.findByRole("option", { name: "alice.smith@example.com" });

    await answerFor("ali", [
      { user_id: "u-jones", user_email: "alice.jones@example.com" },
      { user_id: "u-smith", user_email: "alice.smith@example.com" },
    ]);

    await user.keyboard("{Enter}");
    await user.click(screen.getByRole("button", { name: /add member/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toStrictEqual({
      user_email: "alice.smith@example.com",
      user_id: "u-smith",
      role: "user",
    });
  });

  it("stops loading once the box is cleared and the abandoned search answers", async () => {
    const user = userEvent.setup();
    render(<UserSearchModal isVisible onCancel={vi.fn()} onSubmit={vi.fn()} accessToken="sk-test" />);

    const input = getEmailSearchInput();
    await user.click(input);
    await user.type(input, "ali");
    await waitFor(() => expect(answers.has("ali")).toBe(true), { timeout: 3000 });
    await screen.findByText("Loading...");

    await user.clear(input);
    await screen.findByText("No results");

    await answerFor("ali", [{ user_id: "u-jones", user_email: "alice.jones@example.com" }]);

    expect(screen.queryByRole("option")).not.toBeInTheDocument();
    expect(screen.getByText("No results")).toBeInTheDocument();
  });

  it("keeps loading while a newer search is still in flight", async () => {
    const user = userEvent.setup();
    render(<UserSearchModal isVisible onCancel={vi.fn()} onSubmit={vi.fn()} accessToken="sk-test" />);

    const input = getEmailSearchInput();
    await user.click(input);
    await user.type(input, "ali");
    await waitFor(() => expect(answers.has("ali")).toBe(true), { timeout: 3000 });

    await user.type(input, "ce.smith");
    await waitFor(() => expect(answers.has("alice.smith")).toBe(true), { timeout: 3000 });

    await answerFor("ali", []);

    expect(screen.getByText("Loading...")).toBeInTheDocument();
    expect(screen.queryByText("No results")).not.toBeInTheDocument();

    await answerFor("alice.smith", [{ user_id: "u-smith", user_email: "alice.smith@example.com" }]);
    await screen.findByRole("option", { name: "alice.smith@example.com" });
  });
});

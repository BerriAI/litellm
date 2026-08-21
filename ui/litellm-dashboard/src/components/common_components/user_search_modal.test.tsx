import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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

  it("tells the user that only existing accounts can be selected", () => {
    renderModal();

    const notice = screen.getByRole("alert");
    expect(notice).toHaveTextContent(/users that already exist/i);
    expect(notice).toHaveTextContent(/ask a proxy admin to create their account first/i);
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

  it("submits every registered field, with the untouched identity fields undefined", async () => {
    const { user, onSubmit } = setup();

    await user.click(save());

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const values = onSubmit.mock.calls[0][0];
    expect(Object.keys(values).sort()).toEqual(["role", "user_email", "user_id"]);
    expect(values).toStrictEqual({ user_email: undefined, user_id: undefined, role: "user" });
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

  it("carries a role changed off its default into the payload", async () => {
    const { onSubmit } = setup();
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

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

    await user.type(input, "ce.smith@example.com");
    await waitFor(() => expect(answers.has("alice.smith@example.com")).toBe(true), { timeout: 3000 });

    await answerFor("ali", []);

    expect(screen.getByText("Loading...")).toBeInTheDocument();
    expect(screen.queryByText("No results")).not.toBeInTheDocument();

    await answerFor("alice.smith@example.com", [{ user_id: "u-smith", user_email: "alice.smith@example.com" }]);
    await screen.findByRole("option", { name: "alice.smith@example.com" });
  });
});

import { act, fireEvent, render, screen, within } from "@testing-library/react";
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
    expect(notice.className).toMatch(/ant-alert-info/);
  });
});

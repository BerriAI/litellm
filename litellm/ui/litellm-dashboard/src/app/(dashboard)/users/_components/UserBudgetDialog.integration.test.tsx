import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UserInfo } from "@/components/networking";
import { chooseSelectOption } from "../../../../../tests/test-utils";

import UserBudgetDialog from "./UserBudgetDialog";

const userInfo: Pick<UserInfo, "user_id" | "user_email" | "max_budget" | "budget_duration"> = {
  user_id: "budget-user",
  user_email: "budget@example.test",
  max_budget: 12.345,
  budget_duration: "7d",
};

const fetchMock = vi.fn<typeof fetch>();

const renderDialog = () => {
  const onClose = vi.fn();
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <UserBudgetDialog user={userInfo} accessToken="test-token" onClose={onClose} />
    </QueryClientProvider>,
  );
  return onClose;
};

describe("UserBudgetDialog", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(Response.json({ user_id: userInfo.user_id }));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  it("submits only the selected user's budget and preserves the existing reset schedule", async () => {
    const user = userEvent.setup();
    const onClose = renderDialog();
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(12.345);
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "42.765" } });
    await user.click(screen.getByRole("button", { name: "Save budget" }));

    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/user/update"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ user_id: "budget-user", max_budget: 42.765 }),
      }),
    );
  });

  it("can remove the limit and change its reset period", async () => {
    const user = userEvent.setup();
    const onClose = renderDialog();
    await user.click(screen.getByRole("checkbox", { name: "Unlimited budget" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Reset Budget" }), "monthly");
    await user.click(screen.getByRole("button", { name: "Save budget" }));

    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/user/update"),
      expect.objectContaining({
        body: JSON.stringify({ user_id: "budget-user", max_budget: null, budget_duration: "30d" }),
      }),
    );
  });

  it("keeps the edited values available after an update fails so the admin can retry", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(Response.json({ error: "Update failed" }, { status: 500 }));
    const onClose = renderDialog();
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "25" } });
    await user.click(screen.getByRole("button", { name: "Save budget" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Save budget" })).toBeEnabled());
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(25);
    await user.click(screen.getByRole("button", { name: "Save budget" }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
  });
});

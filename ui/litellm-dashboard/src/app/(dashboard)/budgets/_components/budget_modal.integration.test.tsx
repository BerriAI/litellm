import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import BudgetModal from "./budget_modal";

const { createMock } = vi.hoisted(() => ({ createMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/budgets/useBudgets", () => ({
  useCreateBudget: () => ({ mutateAsync: createMock }),
}));

const FULL_PAYLOAD = {
  budget_id: "budget-alpha",
  tpm_limit: 500.57,
  rpm_limit: 7,
  max_budget: 42.57,
  budget_duration: "30d",
};

const renderModal = () => render(<BudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} />);

const create = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Create Budget" }));

const openOptionalSettings = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByText("Optional Settings"));
  await screen.findByLabelText("Max Budget (USD)");
};

describe("BudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createMock.mockResolvedValue(undefined);
  });

  it("submits only the mounted fields when Optional Settings stays collapsed", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText("Max Tokens per minute"), { target: { value: "500.567" } });
    fireEvent.change(screen.getByLabelText("Max Requests per minute"), { target: { value: "7" } });
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0]).toEqual({
      budget_id: "budget-alpha",
      tpm_limit: 500.57,
      rpm_limit: 7,
    });
  });

  it("submits every field once Optional Settings is expanded", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText("Max Tokens per minute"), { target: { value: "500.567" } });
    fireEvent.change(screen.getByLabelText("Max Requests per minute"), { target: { value: "7" } });

    await openOptionalSettings(user);
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "42.567" } });

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("monthly"));

    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0]).toEqual(FULL_PAYLOAD);
  });

  it("drops Optional Settings values again when the section is collapsed before submit", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });

    await openOptionalSettings(user);
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "42.567" } });
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("monthly"));

    await user.click(screen.getByText("Optional Settings"));
    await waitFor(() => expect(screen.queryByLabelText("Max Budget (USD)")).not.toBeInTheDocument());
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha" });
  });

  it("submits a cleared number field as null", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText("Max Tokens per minute"), { target: { value: "5" } });
    await user.clear(screen.getByLabelText("Max Tokens per minute"));
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0]).toEqual({
      budget_id: "budget-alpha",
      tpm_limit: null,
    });
  });

  it("blocks submit while Budget ID is empty", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Max Tokens per minute"), { target: { value: "5" } });
    await create(user);

    await waitFor(() => expect(screen.getByLabelText("Budget ID")).toHaveAttribute("aria-invalid", "true"));
    expect(createMock).not.toHaveBeenCalled();
  });

  it("keeps a typed Optional Setting when the section is collapsed and reopened, as antd's store did", async () => {
    const user = userEvent.setup();
    renderModal();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "probe-budget" } });

    await openOptionalSettings(user);
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "42.5" } });

    await user.click(screen.getByText("Optional Settings"));
    await user.click(screen.getByText("Optional Settings"));

    expect(await screen.findByLabelText("Max Budget (USD)")).toHaveValue(42.5);

    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0]).toMatchObject({ budget_id: "probe-budget", max_budget: 42.5 });
  });
});

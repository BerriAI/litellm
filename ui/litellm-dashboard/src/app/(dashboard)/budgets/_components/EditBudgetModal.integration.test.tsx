import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@/lib/http/schema";

import EditBudgetModal from "./EditBudgetModal";
import { chooseSelectOption, renderWithProviders as render, testQueryClient } from "../../../../../tests/test-utils";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import { ApiError } from "@/lib/http/client";

vi.unmock("@/lib/toast");

const { updateMock } = vi.hoisted(() => ({ updateMock: vi.fn() }));

// 保留真实预算 hook 和 mutation 生命周期，只替换 HTTP 调用边界。
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  budgetUpdateCall: updateMock,
}));

type BudgetItem = components["schemas"]["BudgetListItem"];

const EXISTING_BUDGET: BudgetItem = {
  budget_id: "budget-alpha",
  max_budget: 12.345,
  budget_duration: "7d",
  tpm_limit: 1000,
  rpm_limit: 10,
  tpd_limit: 50000,
  soft_budget: 25,
  budget_reset_at: "2026-02-01T00:00:00Z",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-02T00:00:00Z",
};

const renderModal = () =>
  render(<EditBudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} existingBudget={EXISTING_BUDGET} />);

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Save" }));

const resetPeriod = () => screen.getByRole("combobox", { name: "Reset period" });

describe("EditBudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    localStorage.clear();
    updateMock.mockResolvedValue(undefined);
  });

  afterEach(() => act(() => toast.dismiss()));

  it("保存前切换语言保留已编辑字段且只提交改动值", async () => {
    render(
      <LanguageProvider defaultLanguage="en" userId="budget-draft" storageScope="test">
        <LanguageSelector />
        <EditBudgetModal isModalVisible setIsModalVisible={vi.fn()} existingBudget={EXISTING_BUDGET} />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText("Requests per minute"), { target: { value: "20" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language", hidden: true }), {
      target: { value: "zh-CN" },
    });
    expect(updateMock).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "保存" }));
    await waitFor(() =>
      expect(updateMock).toHaveBeenCalledExactlyOnceWith("123", { budget_id: "budget-alpha", rpm_limit: 20 }),
    );
  });

  // 语言变化不能清掉 dirtyFields，也不能把既有预算字段重新写回。
  it.each(["success", "empty", "api", "structured", "wrapped", "string"] as const)(
    "预算更新 %s 的通知和错误分类随当前语言显示",
    async (result) => {
      const request = Promise.withResolvers<unknown>();
      updateMock.mockReturnValue(request.promise);
      const setIsModalVisible = vi.fn();
      render(
        <LanguageProvider defaultLanguage="en" userId="edit-budget-notices" storageScope="test">
          <LanguageSelector />
          <EditBudgetModal isModalVisible setIsModalVisible={setIsModalVisible} existingBudget={EXISTING_BUDGET} />
          <Toaster expand />
        </LanguageProvider>,
      );
      fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
      fireEvent.click(screen.getByRole("button", { name: "Save" }));
      await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
      expect(await screen.findByText("Making API Call")).toBeInTheDocument();
      fireEvent.change(screen.getByRole("combobox", { name: "Interface language", hidden: true }), {
        target: { value: "zh-CN" },
      });
      const reasons = {
        empty: new Error(""),
        api: new ApiError("Home", 403, { error: { message: "Home" } }),
        structured: { response: { status: 403, data: { error: { message: "Home" } } } },
        wrapped: new Error(JSON.stringify({ error: { message: "Home" }, internal_debug: "test-only-internal" })),
        string: "Home",
      };
      await act(async () => {
        if (result === "success") request.resolve(undefined);
        else request.reject(reasons[result]);
      });
      if (result === "success") expect(await screen.findByText("预算已更新")).toBeInTheDocument();
      else if (result === "empty") expect(await screen.findByText("更新预算失败，请重试")).toBeInTheDocument();
      else {
        if (result === "api" || result === "structured")
          expect(await screen.findByText("无访问权限")).toBeInTheDocument();
        expect(await screen.findByText("Home", { selector: "[data-description]" })).toBeInTheDocument();
      }
      expect(screen.getByText("正在调用 API")).toBeInTheDocument();
      expect(setIsModalVisible).toHaveBeenCalledTimes(result === "success" ? 1 : 0);
      if (result !== "success") {
        expect(screen.getByLabelText("预算 ID")).toHaveValue("budget-alpha");
        expect(screen.getByLabelText("每分钟令牌数")).toHaveValue(500);
      }
      expect(screen.queryByText(/test-only-internal|\[object Object\]/)).not.toBeInTheDocument();
      fireEvent.change(screen.getByRole("combobox", { name: "界面语言", hidden: true }), { target: { value: "en" } });
      expect(screen.getByText("Making API Call")).toBeInTheDocument();
      if (result === "success") expect(screen.getByText("Budget Updated")).toBeInTheDocument();
      else if (result === "empty") expect(screen.getByText("Failed to update budget")).toBeInTheDocument();
      expect(updateMock).toHaveBeenCalledExactlyOnceWith("123", { budget_id: "budget-alpha", tpm_limit: 500 });
    },
  );

  it("prefills every field from the existing budget", () => {
    renderModal();

    expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(12.345);
    expect(resetPeriod()).toHaveTextContent("per week");
    expect(screen.getByLabelText("Tokens per minute")).toHaveValue(1000);
    expect(screen.getByLabelText("Requests per minute")).toHaveValue(10);
    expect(screen.getByLabelText("Batch tokens per day")).toHaveValue(50000);
  });

  it("sends only the changed rate limit so the stored spend limit and reset period are left as they are", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    expect(updateMock.mock.calls[0][1]).toEqual({ budget_id: "budget-alpha", tpm_limit: 500 });
  });

  it("sends a later change after another field was changed and reverted", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "1000" } });
    fireEvent.change(screen.getByLabelText("Requests per minute"), { target: { value: "20" } });
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    expect(updateMock.mock.calls[0][1]).toEqual({ budget_id: "budget-alpha", rpm_limit: 20 });
  });

  it("submits a changed spend limit and reset period", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, resetPeriod(), "per month");
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    const expected = { budget_id: "budget-alpha", max_budget: 42.57, budget_duration: "30d" };
    expect(updateMock.mock.calls[0][1]).toEqual(expected);
  });

  it("submits a cleared limit and an 'in total' reset period as null", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText("Tokens per minute"));
    await chooseSelectOption(user, resetPeriod(), "in total");
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    const expected = { budget_id: "budget-alpha", tpm_limit: null, budget_duration: null };
    expect(updateMock.mock.calls[0][1]).toEqual(expected);
  });

  it("does not let the budget ID be edited", () => {
    renderModal();

    expect(screen.getByLabelText("Budget ID")).toBeDisabled();
  });
});

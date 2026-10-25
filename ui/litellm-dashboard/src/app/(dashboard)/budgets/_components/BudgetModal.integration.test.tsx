import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import BudgetModal from "./BudgetModal";
import { chooseSelectOption, renderWithProviders as render, testQueryClient } from "../../../../../tests/test-utils";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import { ApiError } from "@/lib/http/client";

vi.unmock("@/lib/toast");

const { createMock } = vi.hoisted(() => ({ createMock: vi.fn() }));

// 保留真实预算 hook 和 mutation 生命周期，只替换 HTTP 调用边界。
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  budgetCreateCall: createMock,
}));

const renderModal = () => render(<BudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} />);

const create = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Create budget" }));

const resetPeriod = () => screen.getByRole("combobox", { name: "Reset period" });

describe("BudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    localStorage.clear();
    createMock.mockResolvedValue(undefined);
  });

  afterEach(() => act(() => toast.dismiss()));

  // 使用真实通知和延迟写入边界，覆盖错误类别、草稿与原始载荷。
  it.each(["success", "empty", "api", "structured", "wrapped", "string"] as const)(
    "预算创建 %s 的通知和错误分类随当前语言显示",
    async (result) => {
      const request = Promise.withResolvers<unknown>();
      createMock.mockReturnValue(request.promise);
      const setIsModalVisible = vi.fn();
      render(
        <LanguageProvider defaultLanguage="en" userId="create-budget-notices" storageScope="test">
          <LanguageSelector />
          <BudgetModal isModalVisible setIsModalVisible={setIsModalVisible} />
          <Toaster expand />
        </LanguageProvider>,
      );
      fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "Home" } });
      fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
      fireEvent.click(screen.getByRole("button", { name: "Create budget" }));
      await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
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
      if (result === "success") expect(await screen.findByText("预算已创建")).toBeInTheDocument();
      else if (result === "empty") expect(await screen.findByText("创建预算失败，请重试")).toBeInTheDocument();
      else {
        if (result === "api" || result === "structured")
          expect(await screen.findByText("无访问权限")).toBeInTheDocument();
        expect(await screen.findByText("Home", { selector: "[data-description]" })).toBeInTheDocument();
      }
      expect(screen.getByText("正在调用 API")).toBeInTheDocument();
      expect(setIsModalVisible).toHaveBeenCalledTimes(result === "success" ? 1 : 0);
      if (result !== "success") {
        expect(screen.getByLabelText("预算 ID")).toHaveValue("Home");
        expect(screen.getByLabelText("每分钟令牌数")).toHaveValue(500);
      }
      expect(screen.queryByText(/test-only-internal|\[object Object\]/)).not.toBeInTheDocument();
      fireEvent.change(screen.getByRole("combobox", { name: "界面语言", hidden: true }), { target: { value: "en" } });
      expect(screen.getByText("Making API Call")).toBeInTheDocument();
      if (result === "success") expect(screen.getByText("Budget Created")).toBeInTheDocument();
      else if (result === "empty") expect(screen.getByText("Failed to create budget")).toBeInTheDocument();
      expect(createMock).toHaveBeenCalledExactlyOnceWith("123", { budget_id: "Home", tpm_limit: 500 });
    },
  );

  // 保留已有错误和数值草稿，语言变化本身不触发预算创建。
  it.each(["en", "zh-CN"] as const)("%s 的已显示预算 ID 错误随语言切换", async (initial) => {
    render(
      <LanguageProvider defaultLanguage={initial} userId="budget-validation" storageScope="test">
        <LanguageSelector />
        <BudgetModal isModalVisible setIsModalVisible={vi.fn()} />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText(initial === "en" ? "Tokens per minute" : "每分钟令牌数"), {
      target: { value: "500" },
    });
    fireEvent.click(screen.getByRole("button", { name: initial === "en" ? "Create budget" : "创建预算" }));
    const original = initial === "en" ? "Enter a budget ID" : "请输入预算 ID";
    const next = initial === "en" ? "请输入预算 ID" : "Enter a budget ID";
    expect(await screen.findByText(original)).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: /Interface language|界面语言/, hidden: true }), {
      target: { value: initial === "en" ? "zh-CN" : "en" },
    });
    expect(screen.getByText(next)).toBeInTheDocument();
    expect(screen.queryByText(original)).not.toBeInTheDocument();
    expect(screen.getByLabelText(initial === "en" ? "每分钟令牌数" : "Tokens per minute")).toHaveValue(500);
    expect(createMock).not.toHaveBeenCalled();
  });

  it("submits the spend limit and reset period alongside the rate limits", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(screen.getByLabelText(/^Spend limit/)).toBeVisible();
    expect(resetPeriod()).toBeVisible();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, resetPeriod(), "per month");
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    fireEvent.change(screen.getByLabelText("Requests per minute"), { target: { value: "7" } });
    fireEvent.change(screen.getByLabelText("Batch tokens per day"), { target: { value: "90000" } });
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "budget-alpha",
      max_budget: 42.57,
      budget_duration: "30d",
      tpm_limit: 500,
      rpm_limit: 7,
      tpd_limit: 90000,
    };
    expect(createMock.mock.calls[0][1]).toEqual(expected);
  });

  it("submits a spend limit that never resets as a null reset period", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "10" } });
    expect(resetPeriod()).toHaveTextContent("in total");
    await chooseSelectOption(user, resetPeriod(), "per week");
    await chooseSelectOption(user, resetPeriod(), "in total");
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][1]).toEqual({ budget_id: "budget-alpha", max_budget: 10, budget_duration: null });
  });

  it("submits a cleared number field as null", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "5" } });
    await user.clear(screen.getByLabelText("Tokens per minute"));
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][1]).toEqual({ budget_id: "budget-alpha", tpm_limit: null });
  });

  it("blocks submit while Budget ID is empty", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "5" } });
    await create(user);

    await waitFor(() => expect(screen.getByLabelText("Budget ID")).toHaveAttribute("aria-invalid", "true"));
    expect(createMock).not.toHaveBeenCalled();
  });
});

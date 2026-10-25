import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import BudgetModal from "@/app/(dashboard)/budgets/_components/BudgetModal";
import ResetMemberBudgetsDialog from "@/components/team/ResetMemberBudgetsDialog";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { chooseSelectOption } from "../../tests/test-utils";

const { createBudget } = vi.hoisted(() => ({ createBudget: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/budgets/useBudgets", () => ({
  useCreateBudget: () => ({ mutateAsync: createBudget }),
}));

function SwitchLanguage() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>Switch to English</button>;
}

describe("管理页语言切换", () => {
  beforeEach(() => {
    localStorage.clear();
    createBudget.mockReset().mockResolvedValue(undefined);
  });

  it("翻译预算周期和校验提示，提交时保留预算 ID 与周期值", async () => {
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="management-test" storageScope="management">
        <BudgetModal isModalVisible setIsModalVisible={vi.fn()} />
      </LanguageProvider>,
    );

    await user.click(screen.getByRole("button", { name: "创建预算" }));
    expect(await screen.findByText("请输入预算 ID")).toBeVisible();
    expect(createBudget).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("预算 ID"), { target: { value: "Home" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "重置周期" }), "每月");
    fireEvent.change(screen.getByLabelText("每分钟请求数"), { target: { value: "7" } });
    await user.click(screen.getByRole("button", { name: "创建预算" }));

    await waitFor(() =>
      expect(createBudget).toHaveBeenCalledWith({ budget_id: "Home", budget_duration: "30d", rpm_limit: 7 }),
    );
  });

  it("将重置预算说明作为整句翻译，切回英文后保留金额和操作", async () => {
    const user = userEvent.setup();
    const reset = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="dialog-test" storageScope="management">
        <ResetMemberBudgetsDialog
          state={{
            phase: "prompting",
            pending: { teamId: "Home", updateData: {}, userIds: ["Home", "Settings"], newBudget: 1000 },
          }}
          onReset={reset}
          onRetry={vi.fn()}
          onKeep={vi.fn()}
          onDismiss={vi.fn()}
        />
        <SwitchLanguage />
      </LanguageProvider>,
    );

    expect(
      screen.getByText(
        "2 位成员设置了自定义预算，因此新的团队默认预算 $1,000.00 不会应用于这些成员。要重置为默认值，还是保留自定义预算？",
      ),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "全部重置为 $1,000.00" }));
    expect(reset).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "Switch to English", hidden: true }));
    expect(screen.getByRole("button", { name: "Reset all to $1,000.00" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Keep custom budgets" })).toBeVisible();
  });
});

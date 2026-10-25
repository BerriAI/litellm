import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { chooseSelectOption, renderWithProviders, screen } from "../../../tests/test-utils";
import { LanguageProvider } from "@/contexts/LanguageContext";
import EntityUsageExportModal from "./EntityUsageExportModal";
import UsageExportHeader from "./UsageExportHeader";

vi.mock("./utils", () => ({
  downloadBlob: vi.fn(),
  exportFilename: vi.fn(() => "team_usage_daily_with_models.json"),
}));

const dateRange = { from: new Date(2026, 9, 1), to: new Date(2026, 9, 14) };

describe("实体用量导出汉化", () => {
  beforeEach(() => localStorage.clear());

  it("汉化类型、格式和计数，保持导出参数不变", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn().mockResolvedValue(new Blob(["data"]));
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="export-test" storageScope="export-test">
        <EntityUsageExportModal
          isOpen
          onClose={vi.fn()}
          entityType="team"
          onExport={onExport}
          dateRange={dateRange}
          selectedFilters={["Home", "Settings"]}
        />
      </LanguageProvider>,
    );
    expect(screen.getByRole("heading", { name: "导出团队用量" })).toBeInTheDocument();
    expect(screen.getByText("格式")).toBeInTheDocument();
    expect(screen.getByText("导出类型")).toBeInTheDocument();
    expect(screen.getByText(/2 个筛选条件/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "取消" })).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /按团队和模型逐日统计/ }));
    await chooseSelectOption(user, screen.getByRole("combobox"), "JSON（包含元数据）");
    await user.click(screen.getByRole("button", { name: "导出 JSON" }));
    expect(onExport).toHaveBeenCalledWith("daily_with_models", "json");
  });

  it("汉化导出入口和空筛选说明，不翻译调用者的标题或筛选选项", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="export-test" storageScope="export-test">
        <UsageExportHeader
          dateValue={dateRange}
          entityType="tag"
          onExport={vi.fn()}
          showFilters
          filterLabel="Home"
          filterOptions={[]}
          customTitle="Settings"
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("此时间范围内没有用量的标签")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "导出数据" }));
    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument();
  });
});

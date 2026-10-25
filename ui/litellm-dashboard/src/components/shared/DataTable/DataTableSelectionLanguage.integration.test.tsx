import type { ColumnDef } from "@tanstack/react-table";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { createSelectionColumn, DataTable } from "./index";

const data = [{ id: "row-1", name: "Home" }];
const nameColumn: ColumnDef<(typeof data)[number]> = { id: "name", accessorKey: "name", header: "Name" };

// 默认无障碍标签随上下文翻译，自定义业务标签和行数据不参与翻译。
describe("表格选择列默认标签语言", () => {
  beforeEach(() => localStorage.clear());

  it("未传入自定义标签时在中英文间切换且保留选择状态", () => {
    const columns = [createSelectionColumn<(typeof data)[number]>(), nameColumn];
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="table-selection" storageScope="test">
        <DataTable data={data} columns={columns} getRowId={(row) => row.id} />
        <LanguageSelector />
      </LanguageProvider>,
    );
    fireEvent.click(screen.getByRole("checkbox", { name: "选择此行" }));
    expect(screen.getByRole("checkbox", { name: "选择此行" })).toBeChecked();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByRole("checkbox", { name: "Select row" })).toBeChecked();
    expect(screen.getByText("Home")).toBeVisible();
  });

  it("调用方提供的标签即使匹配词条也保留原文", () => {
    const columns = [createSelectionColumn<(typeof data)[number]>({ rowAriaLabel: () => "Settings" }), nameColumn];
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="table-selection" storageScope="test">
        <DataTable data={data} columns={columns} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("checkbox", { name: "Settings" })).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
  });
});

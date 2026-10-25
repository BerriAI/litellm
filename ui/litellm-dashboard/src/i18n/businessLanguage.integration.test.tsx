import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { DataTablePagination } from "@/components/shared/DataTable/DataTablePagination";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { PaginatedMultiSelect } from "@/components/shared/PaginatedMultiSelect";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import DocumentsTable from "@/app/(dashboard)/vector-stores/_components/DocumentsTable";
import { BarChart } from "@/components/shared/charts/bar_chart";
import { AreaChart } from "@/components/shared/charts/area_chart";
import { CustomTooltip } from "@/components/shared/charts/chart_tooltip";
import BudgetDurationDropdown, {
  getBudgetDurationLabel,
} from "@/components/common_components/budget_duration_dropdown";
import { translate } from "@/i18n/language";
import {
  StreamScopeFields,
  GuardrailStreamScopeDetail,
} from "@/app/(dashboard)/guardrails/_components/StreamScopeFields";
import ImpactPreviewAlert from "@/app/(dashboard)/policies/_components/impact_preview_alert";
import ModelFilters from "@/components/model_filters";

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

describe("业务组件语言切换", () => {
  beforeAll(() => {
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) =>
      setTimeout(() => callback({ didTimeout: false, timeRemaining: () => 50 }), 0),
    );
  });
  afterAll(() => vi.unstubAllGlobals());
  it.each([
    ["dashboard", "Home"],
    ["chat", "Home"],
    ["dashboard", "Vision"],
    ["chat", "Vision"],
  ] as const)("%s 模型筛选保留未知模式 %s 和扩展能力原文", async (variant, modeValue) => {
    const user = userEvent.setup();
    const onModeChange = vi.fn();
    const onFilteredDataChange = vi.fn();
    const model = {
      model_group: "Settings",
      providers: [],
      mode: modeValue,
      supports_settings: true,
      supports_vision: true,
      supports_function_calling: false,
      supports_parallel_function_calling: false,
      is_public_model_group: true,
    };
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId={`filters-${variant}`} storageScope="business-test">
        <ModelFilters
          variant={variant}
          showProviderFilter={false}
          modelHubData={[model]}
          onModeChange={onModeChange}
          onFilteredDataChange={onFilteredDataChange}
        />
      </LanguageProvider>,
    );

    if (variant === "chat") await user.click(screen.getByRole("combobox", { name: "模型类型" }));
    const mode = await screen.findByRole("option", { name: modeValue });
    if (variant === "chat") await user.click(mode);
    else fireEvent.change(screen.getAllByRole("combobox")[0], { target: { value: modeValue } });
    expect(onModeChange).toHaveBeenCalledWith(modeValue);
    expect(onFilteredDataChange).toHaveBeenLastCalledWith([model]);

    if (variant === "chat") await user.click(screen.getByRole("combobox", { name: "模型能力" }));
    expect(await screen.findByRole("option", { name: "Settings" })).toBeVisible();
    expect(screen.getByRole("option", { name: "视觉" })).toBeVisible();
    expect(screen.queryByRole("option", { name: "设置" })).not.toBeInTheDocument();
  });
  it("流式范围选中项与详情可切换语言，回调保留原始模式值", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="stream-scope" storageScope="business-test">
        <LanguageSwitch />
        <StreamScopeFields modes={["post_call"]} value={{ post_call: "both" }} onChange={onChange} />
        <GuardrailStreamScopeDetail raw={{ post_call: "streaming" }} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox")).toHaveTextContent("流式和非流式");
    expect(screen.getByText("post_call: 仅流式")).toBeInTheDocument();
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "仅非流式" }));
    expect(onChange).toHaveBeenCalledWith({ post_call: "non_streaming" });
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("combobox")).toHaveTextContent("Streaming and non-streaming");
  });

  it("预算周期选中值已翻译，选择后仍传原始周期，未知周期不翻译", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="budget-period" storageScope="business-test">
        <BudgetDurationDropdown value="30d" onChange={onChange} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox")).toHaveTextContent("每月");
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "每周" }));
    expect(onChange).toHaveBeenCalledWith("7d", expect.anything());
    expect(getBudgetDurationLabel("Home", (message, values) => translate("zh-CN", message, values))).toBe("Home");
  });

  it("影响预览翻译完整计数句子且不翻译样例名称", () => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="impact-count" storageScope="business-test">
        <LanguageSwitch />
        <ImpactPreviewAlert
          isDefault
          impactResult={{ affected_keys_count: 7, affected_teams_count: 1, sample_keys: ["Home"], sample_teams: [] }}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("此关联项最多会影响 7 个密钥和 1 个团队。")).toBeInTheDocument();
    expect(screen.getByText("另有 2 项…")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText("This attachment would affect up to 7 keys and 1 team.")).toBeInTheDocument();
  });
  it("分页显示完整中文句子，操作仍传原始页码，英文空格保持不变", () => {
    const onPageChange = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="pagination" storageScope="business-test">
        <LanguageSwitch />
        <DataTablePagination
          page={1}
          pageSize={25}
          rowCount={100}
          onPageChange={onPageChange}
          onPageSizeChange={vi.fn()}
        />
      </LanguageProvider>,
    );
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("第 26–50 条，共 100 条");
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("第 2 页，共 4 页");
    fireEvent.click(screen.getByRole("button", { name: "转到下一页" }));
    expect(onPageChange).toHaveBeenCalledWith(2);
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 26-50 of 100");
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 4");
  });

  it("翻译默认占位符，但不翻译与词典键同名的业务选项", async () => {
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="options" storageScope="business-test">
        <SearchSelect options={[{ label: "Home", value: "model-original-id" }]} onValueChange={vi.fn()} />
      </LanguageProvider>,
    );
    expect(screen.getByPlaceholderText("请选择…")).toBeInTheDocument();
    await user.click(screen.getByRole("combobox"));
    await user.keyboard("{ArrowDown}");
    expect(await screen.findByRole("option", { name: "Home" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "首页" })).not.toBeInTheDocument();
  });

  it("文档表头和受控状态跟随语言切换，文件名和删除ID保持原样", async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="documents" storageScope="business-test">
        <LanguageSwitch />
        <DocumentsTable documents={[{ uid: "file-original-id", name: "Home", status: "done" }]} onRemove={onRemove} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("columnheader", { name: "名称" })).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("就绪")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "打开文档操作菜单" }));
    await user.click(await screen.findByRole("menuitem", { name: "移除" }));
    expect(onRemove).toHaveBeenCalledWith("file-original-id");
    await user.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("columnheader", { name: "Name" })).toBeInTheDocument();
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
  });

  it("多选框翻译默认提示和自定义创建说明但不改业务值", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="multiselect" storageScope="business-test">
        <MultiSelect options={[]} onValueChange={onValueChange} allowCustomValues />
      </LanguageProvider>,
    );
    await user.click(screen.getByRole("combobox"));
    await user.type(screen.getByRole("combobox"), "Home");
    await user.click(await screen.findByRole("option", { name: "创建“Home”" }));
    expect(onValueChange).toHaveBeenCalledWith(["Home"]);
  });

  it("分页多选默认提示跟随语言，不翻译用户提供的错误信息", async () => {
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="paginated" storageScope="business-test">
        <LanguageSwitch />
        <PaginatedMultiSelect
          options={[]}
          onValueChange={vi.fn()}
          onSearchChange={vi.fn()}
          onLoadMore={vi.fn()}
          errorText="Home"
        />
      </LanguageProvider>,
    );
    expect(screen.getByPlaceholderText("搜索…")).toBeInTheDocument();
    await user.click(screen.getByRole("combobox"));
    await user.keyboard("{ArrowDown}");
    expect(await screen.findByText("Home")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByPlaceholderText("Search…")).toBeInTheDocument();
  });

  it("中文日期预设仍返回原始shortLabel并支持英文切换", async () => {
    const onValueChange = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="date" storageScope="business-test">
        <LanguageSwitch />
        <AdvancedDatePicker
          value={{ from: new Date(2026, 0, 1), to: new Date(2026, 0, 31) }}
          onValueChange={onValueChange}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("选择时间范围")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { expanded: false }));
    fireEvent.click(screen.getByRole("button", { name: /最近 7 天/ }));
    fireEvent.click(screen.getByRole("button", { name: "应用" }));
    expect(onValueChange).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ from: expect.any(Date), to: expect.any(Date) }),
      "7d",
    );
    await waitFor(() => expect(onValueChange).toHaveBeenNthCalledWith(2, expect.anything(), "7d"));
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText("Select Time Range")).toBeInTheDocument();
  });

  it("图表图例使用展示别名但不翻译原始分类或未知提示指标", () => {
    const { rerender } = render(
      <BarChart
        data={[{ model: "Home", spend: 3 }]}
        index="model"
        categories={["spend"]}
        categoryLabels={{ spend: "费用" }}
      />,
    );
    expect(screen.getByText("费用")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    rerender(
      <AreaChart
        data={[{ model: "Home", spend: 3 }]}
        index="model"
        categories={["spend"]}
        categoryLabels={{ spend: "费用" }}
      />,
    );
    expect(screen.getByText("费用")).toBeInTheDocument();
    rerender(
      <LanguageProvider defaultLanguage="zh-CN" userId="tooltip" storageScope="business-test">
        <CustomTooltip active payload={[{ dataKey: "metrics.Home", value: 3, payload: { metrics: { Home: 3 } } }]} />
      </LanguageProvider>,
    );
    expect(screen.getByText("Home")).toBeInTheDocument();
  });
});

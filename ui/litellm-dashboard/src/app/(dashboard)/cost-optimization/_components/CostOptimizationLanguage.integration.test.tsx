import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import PromptCachingTab from "./PromptCachingTab";
import PromptCompressionTab from "./PromptCompressionTab";
import { toast } from "@/lib/toast";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";

vi.unmock("@/lib/toast");

const getSettings = vi.fn();
const getGuardrails = vi.fn();

vi.mock("@/components/networking", () => ({
  getGeneralSettingsCall: (...args: unknown[]) => getSettings(...args),
  getGuardrailsList: (...args: unknown[]) => getGuardrails(...args),
  createGuardrailCall: vi.fn(),
}));
vi.mock("@/app/(dashboard)/router-settings/_components/general_settings", () => ({
  PromptCachingPanel: ({
    settings,
    onChange,
  }: {
    settings: { field_name: string; field_value: string }[];
    onChange: (field: string, value: unknown) => void;
  }) => (
    <input
      aria-label="cache draft"
      value={settings[0]?.field_value ?? ""}
      onChange={(event) => onChange("cache", event.target.value)}
    />
  ),
}));
vi.mock("./CacheLeakageCard", () => ({ default: () => null }));
vi.mock("./PromptCachingRequestsTable", () => ({ default: () => null }));
vi.mock("@/components/shared/advanced_date_picker", () => ({ default: () => null }));

const activity = {
  dateValue: {},
  onDateChange: vi.fn(),
  results: [],
  metadata: EMPTY_DAILY_ACTIVITY_METADATA,
  loading: false,
  failed: false,
  scope: { accessToken: "test-token", startTime: null, endTime: null, userId: null, apiKey: null },
};

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("zh-CN")}>切换语言</button>;
}

describe("费用优化语言切换", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
    getSettings.mockClear().mockResolvedValue([{ field_name: "cache", field_value: "server" }]);
    getGuardrails.mockClear().mockResolvedValue({ guardrails: [] });
  });

  afterEach(() => act(() => toast.dismiss()));

  it("缓存设置加载失败后切换语言更新现有通知且不重载", async () => {
    getSettings.mockRejectedValue(new Error("Home"));
    render(
      <LanguageProvider defaultLanguage="en" userId="cache-notice-switch" storageScope="cache-notice-switch">
        <LanguageSelector />
        <PromptCachingTab accessToken="test-token" activity={activity} />
        <Toaster expand />
      </LanguageProvider>,
    );
    expect(await screen.findByText("Failed to load prompt caching settings")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(await screen.findByText("提示词缓存设置加载失败")).toBeVisible();
    expect(screen.queryByText("Failed to load prompt caching settings")).not.toBeInTheDocument();
    expect(getSettings).toHaveBeenCalledTimes(1);
  });

  it("先发出的加载请求失败时，错误提示使用切换后的语言", async () => {
    const user = userEvent.setup();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    let rejectSettings!: (error: Error) => void;
    let rejectGuardrails!: (error: Error) => void;
    getSettings.mockReturnValue(
      new Promise((_, reject) => {
        rejectSettings = reject;
      }),
    );
    getGuardrails.mockReturnValue(
      new Promise((_, reject) => {
        rejectGuardrails = reject;
      }),
    );
    render(
      <LanguageProvider defaultLanguage="en" userId="cost-error-test" storageScope="cost-error-test">
        <LanguageSwitch />
        <PromptCachingTab accessToken="test-token" activity={activity} />
        <PromptCompressionTab accessToken="test-token" />
        <Toaster expand />
      </LanguageProvider>,
    );
    try {
      await user.click(screen.getByRole("button", { name: "切换语言" }));
      await act(async () => {
        rejectSettings(new Error("test settings failure"));
        rejectGuardrails(new Error("test guardrails failure"));
      });
      expect(await screen.findByText("提示词缓存设置加载失败")).toBeVisible();
      expect(screen.getByText("压缩防护规则加载失败")).toBeVisible();
      expect(getSettings).toHaveBeenCalledTimes(1);
      expect(getGuardrails).toHaveBeenCalledTimes(1);
    } finally {
      consoleError.mockRestore();
    }
  });

  it("切换语言不重新加载设置或护栏，也不覆盖未保存草稿", async () => {
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="en" userId="cost-draft-test" storageScope="cost-draft-test">
        <LanguageSwitch />
        <PromptCachingTab accessToken="test-token" activity={activity} />
        <PromptCompressionTab accessToken="test-token" />
      </LanguageProvider>,
    );
    await waitFor(() => expect(screen.getByLabelText("cache draft")).toHaveValue("server"));
    fireEvent.change(screen.getByLabelText("cache draft"), { target: { value: "unsaved cache" } });
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "unsaved guardrail" } });
    await user.click(screen.getByRole("button", { name: "切换语言" }));
    await screen.findByLabelText("名称");
    expect(getSettings).toHaveBeenCalledTimes(1);
    expect(getGuardrails).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText("cache draft")).toHaveValue("unsaved cache");
    expect(screen.getByLabelText("名称")).toHaveValue("unsaved guardrail");
  });
});

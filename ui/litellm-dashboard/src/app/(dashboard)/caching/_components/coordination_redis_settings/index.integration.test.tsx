import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import CoordinationRedisSettings from "./index";
import * as networking from "@/components/networking";
import { toast } from "@/lib/toast";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";

vi.mock("@/components/networking", () => ({
  getCoordinationRedisSettingsCall: vi.fn(),
  testCoordinationRedisConnectionCall: vi.fn(),
  updateCoordinationRedisSettingsCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

const getSettings = vi.mocked(networking.getCoordinationRedisSettingsCall);
const updateSettings = vi.mocked(networking.updateCoordinationRedisSettingsCall);
const testConnection = vi.mocked(networking.testCoordinationRedisConnectionCall);
const notifications = vi.mocked(toast);

const settingsResponse = (
  values: Record<string, unknown>,
  source: "coordination_redis" | "cache_backend" | "environment" | null = null,
) => ({ values, fields: [], source });

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
};

const renderSettings = () => render(<CoordinationRedisSettings />, { wrapper });

const clickSave = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: /save changes/i }));
describe("CoordinationRedisSettings value retention across redis types", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    getSettings.mockResolvedValue(settingsResponse({}));
    updateSettings.mockResolvedValue(undefined);
  });

  const pickRedisType = async (user: ReturnType<typeof userEvent.setup>, name: RegExp) => {
    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click(await screen.findByRole("option", { name }));
  };

  // 不重新校验或重置表单，仅随语言变化更新已显示的本地错误。
  it("切换语言即时更新校验错误且保留协调 Redis 草稿", async () => {
    getSettings.mockResolvedValue(settingsResponse({ host: "redis.internal" }));
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="coordination-validation" storageScope="test">
        <LanguageSelector />
        <CoordinationRedisSettings />
      </LanguageProvider>,
      { wrapper },
    );
    await screen.findByDisplayValue("redis.internal");
    fireEvent.change(screen.getByLabelText("端口"), { target: { value: "99999" } });
    fireEvent.click(screen.getByRole("button", { name: "测试连接" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("端口必须是 1 到 65535 之间的整数");

    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByRole("alert")).toHaveTextContent("Port must be an integer between 1 and 65535");
    expect(screen.getByLabelText("Port")).toHaveValue("99999");
    expect(screen.getByLabelText("Port")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByLabelText("Host")).toHaveValue("redis.internal");
    expect(getSettings).toHaveBeenCalledTimes(1);
    expect(updateSettings).not.toHaveBeenCalled();
    expect(testConnection).not.toHaveBeenCalled();
  });

  it.each([
    [
      "coordination_redis",
      "Configured here",
      "在此配置",
      "已设置 general_settings.coordination_redis，协调功能使用独立的 Redis 连接。",
    ],
    [
      "cache_backend",
      "Borrowed from response cache",
      "复用响应缓存",
      "未配置协调 Redis，代理复用响应缓存的 Redis 连接。",
    ],
    [
      "environment",
      "From REDIS_* environment",
      "来自 REDIS_* 环境变量",
      "未配置协调 Redis，代理回退到 REDIS_* 环境变量。",
    ],
    [null, "Not configured", "未配置", "跨节点限流、费用跟踪和节点锁管理器没有可用于协调的 Redis。"],
  ] as const)("配置来源 %s 的标签和说明随界面语言变化", async (source, en, zh, detail) => {
    getSettings.mockResolvedValue(settingsResponse({ host: "redis.internal" }, source));
    render(
      <LanguageProvider defaultLanguage="en" userId="coordination-source" storageScope="test">
        <LanguageSelector />
        <CoordinationRedisSettings />
      </LanguageProvider>,
      { wrapper },
    );
    expect(await screen.findByTestId("coordination-redis-source")).toHaveTextContent(en);
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByTestId("coordination-redis-source")).toHaveTextContent(zh);
    expect(screen.getByText(detail)).toBeVisible();
    expect(getSettings).toHaveBeenCalledTimes(1);
  });

  it("keeps a value typed into a sentinel-only field when the type is switched away and back", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText("Host");

    await pickRedisType(user, /sentinel/i);
    fireEvent.change(await screen.findByLabelText("Service Name"), { target: { value: "mymaster" } });

    await pickRedisType(user, /node/i);
    await waitFor(() => expect(screen.queryByLabelText("Service Name")).not.toBeInTheDocument());

    await pickRedisType(user, /sentinel/i);

    expect(await screen.findByLabelText("Service Name")).toHaveValue("mymaster");
  });

  it("leaves a sentinel-only value out of the payload once the type is no longer sentinel", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText("Host");

    await pickRedisType(user, /sentinel/i);
    fireEvent.change(await screen.findByLabelText("Service Name"), { target: { value: "mymaster" } });

    await pickRedisType(user, /node/i);
    await waitFor(() => expect(screen.queryByLabelText("Service Name")).not.toBeInTheDocument());

    await clickSave(user);

    await waitFor(() => expect(updateSettings).toHaveBeenCalled());
    expect(JSON.stringify(updateSettings.mock.calls[0])).not.toContain("mymaster");
  });
});

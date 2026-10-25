import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import IndexesTab from "@/app/(dashboard)/vector-stores/_components/IndexesTab";
import LanguageSelector from "@/components/LanguageSelector";
import Settings from "@/components/settings";
import CacheSettings from "@/app/(dashboard)/caching/_components/cache_settings";
import CoordinationRedisSettings from "@/app/(dashboard)/caching/_components/coordination_redis_settings";
import ChangePasswordForm from "@/app/(dashboard)/change-password/ChangePasswordForm";
import { Toaster } from "@/components/ui/sonner";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { toast } from "@/lib/toast";
import { translate } from "@/i18n/language";
import { renderWithProviders, testQueryClient } from "../../tests/test-utils";

vi.unmock("@/lib/toast");

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token" }),
}));

// 按场景提供网络响应，测试主体只负责实际语言切换流程。
const redisResponse = (scenario: string, target: "cache" | "coordination_redis") => {
  if (scenario === "test-unknown") return Response.json({ status: "unhealthy" });
  if (scenario === "test-error") {
    return Response.json({ status: "unhealthy", message: "Settings", error: "Settings" });
  }
  if (scenario.endsWith("error") || scenario === "test-rejected") {
    return Response.json({ error: { message: "Settings" } }, { status: 400 });
  }
  return Response.json({ status: target === "cache" ? "success" : "healthy" });
};

// 请求保持等待时切换语言，只替换网络边界，验证真实组件和通知的上下文连接。
describe("异步错误提示语言", () => {
  beforeEach(() => {
    localStorage.clear();
    testQueryClient.clear();
  });

  afterEach(() => {
    act(() => toast.dismiss());
    vi.restoreAllMocks();
  });

  it.each(["success", "error"] as const)("改密码 %s 响应完成前后切换语言不重发请求", async (scenario) => {
    let finishRequest: (response: Response) => void = () => undefined;
    const pending = new Promise<Response>((resolve) => {
      finishRequest = resolve;
    });
    const requests: { path: string; body: unknown }[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = new URL(String(input), "http://localhost");
      requests.push({ path: url.pathname, body: init?.body });
      if (url.pathname === "/user/password/change") return pending;
      throw new Error(`未配置测试响应 ${url}`);
    });
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="password-notification" storageScope="test">
        <LanguageSelector />
        <Toaster expand />
        <ChangePasswordForm />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText("Current Password"), { target: { value: "Settings" } });
    fireEvent.change(screen.getByLabelText("New Password"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("Confirm New Password"), { target: { value: "Home" } });
    fireEvent.click(screen.getByRole("button", { name: "Change Password" }));
    await waitFor(() => expect(requests).toHaveLength(1));
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    await act(async () => {
      finishRequest(
        scenario === "success"
          ? Response.json({ user_id: "test-user", message: "Password updated successfully." })
          : Response.json({ error: { message: "New passwords do not match" } }, { status: 400 }),
      );
    });
    const message = scenario === "success" ? translate("zh-CN", "Password updated") : "New passwords do not match";
    expect(await screen.findByText(message)).toBeVisible();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(await screen.findByText(scenario === "success" ? "Password updated" : message)).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(JSON.parse(String(requests[0].body))).toStrictEqual({ current_password: "Settings", new_password: "Home" });
    if (scenario === "error") {
      expect(screen.getByLabelText("Current Password")).toHaveValue("Settings");
      expect(screen.getByLabelText("New Password")).toHaveValue("Home");
    }
  });

  it.each([
    {
      target: "indexes",
      path: "/v1/indexes",
      zh: "获取索引失败：Home",
      en: "Error fetching indexes: Home",
      error: new Error("Home"),
    },
    {
      target: "callbacks",
      path: "/callbacks/configs",
      zh: "加载回调配置失败：Home",
      en: "Failed to load callback configs: Home",
      error: new Error("Home"),
    },
    {
      target: "indexes",
      path: "/v1/indexes",
      zh: "获取索引失败：Home",
      en: "Error fetching indexes: Home",
      error: new Error(JSON.stringify({ error: { message: "Home", internal: "internal implementation detail" } })),
    },
  ])("$target 请求失败后使用当前语言，不重发请求或翻译后端详情", async ({ target, path, zh, en, error }) => {
    let rejectRequest: (reason: Error) => void = () => undefined;
    const pending = new Promise<Response>((_resolve, reject) => {
      rejectRequest = reject;
    });
    const requests: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requests.push(url);
      if (url.endsWith(path)) return pending;
      if (url.endsWith("/get/config/callbacks")) {
        return Response.json({ callbacks: [], available_callbacks: {}, alerts: [] });
      }
      if (url.endsWith("/cloudzero/settings")) return Response.json(null);
      if (url.endsWith("/alerting/settings")) return Response.json([]);
      if (url.endsWith("/email/event_settings")) return Response.json({ settings: [] });
      throw new Error(`未配置测试响应 ${url}`);
    });

    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="async-language" storageScope="test">
        <LanguageSelector />
        <Toaster expand />
        {target === "indexes" ? (
          <IndexesTab accessToken="test-token" vectorStores={[]} onViewVectorStore={vi.fn()} />
        ) : (
          <Settings accessToken="test-token" userID="user-1" userRole="Admin" premiumUser={false} />
        )}
      </LanguageProvider>,
    );

    expect(requests.filter((url) => url.endsWith(path))).toHaveLength(1);
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    await act(async () => rejectRequest(error));
    expect(await screen.findByText(zh)).toBeVisible();
    expect(screen.queryByText(/首页/)).not.toBeInTheDocument();
    expect(screen.queryByText(/internal implementation detail/)).not.toBeInTheDocument();
    expect(requests.filter((url) => url.endsWith(path))).toHaveLength(1);

    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(await screen.findByText(en)).toBeVisible();
    expect(requests.filter((url) => url.endsWith(path))).toHaveLength(1);
  });

  // 请求等待和通知已显示两个阶段都切换语言，不修改保存载荷或重发操作。
  describe.each(["cache", "coordination_redis"] as const)("%s 设置通知", (target) => {
    const messages = {
      "test-error": "Connection test failed: Settings",
      "test-rejected": "Connection test failed: Settings",
      "test-unknown": "Connection test failed: Unknown error",
      "test-success":
        target === "cache" ? "Cache connection test successful!" : "Coordination Redis connection test successful!",
      "save-success":
        target === "cache"
          ? "Cache settings updated successfully"
          : "Coordination Redis settings saved. Restart the proxy to apply them.",
      "save-error":
        target === "cache" ? "Failed to update cache settings" : "Failed to update coordination Redis settings",
      "load-error": target === "cache" ? "Failed to load cache settings" : "Failed to load coordination Redis settings",
    };
    it.each([
      "test-error",
      "test-unknown",
      "test-rejected",
      "test-success",
      "save-error",
      "save-success",
      "load-error",
    ] as const)("%s 在请求完成前后都使用当前语言且保留服务器详情", async (scenario) => {
      let finishRequest: (response: Response) => void = () => undefined;
      const pending = new Promise<Response>((resolve) => {
        finishRequest = resolve;
      });
      const basePath = `/${target}/settings`;
      const path = scenario.startsWith("test") ? `${basePath}/test` : basePath;
      const requests: { path: string; method: string; body: unknown }[] = [];
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
        const url = new URL(String(input), "http://localhost");
        const method = init?.method ?? "GET";
        requests.push({ path: url.pathname, method, body: init?.body });
        if (url.pathname === "/model_group/info") return Response.json({ data: [] });
        if (url.pathname === path && (method === "POST" || scenario === "load-error")) return pending;
        if (url.pathname === basePath) {
          return Response.json(
            target === "cache"
              ? { current_values: { host: "redis.internal" } }
              : { values: { host: "redis.internal" }, fields: [], source: null },
          );
        }
        throw new Error(`未配置测试响应 ${url}`);
      });
      renderWithProviders(
        <LanguageProvider defaultLanguage="en" userId="redis-notification" storageScope="test">
          <LanguageSelector />
          <Toaster expand />
          {target === "cache" ? (
            <CacheSettings accessToken="test-token" userRole="Admin" userID="user-1" />
          ) : (
            <CoordinationRedisSettings />
          )}
        </LanguageProvider>,
      );

      if (scenario !== "load-error") {
        await screen.findByDisplayValue("redis.internal");
        fireEvent.change(screen.getByLabelText("Host"), { target: { value: "draft.internal" } });
        fireEvent.click(
          screen.getByRole("button", { name: scenario.startsWith("test") ? "Test Connection" : "Save Changes" }),
        );
        await waitFor(() => expect(requests.filter((request) => request.method === "POST")).toHaveLength(1));
      }
      fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
      await act(async () => finishRequest(redisResponse(scenario, target)));

      const en = messages[scenario];
      const zh =
        en === "Connection test failed: Settings"
          ? translate("zh-CN", "Connection test failed: {error}", { error: "Settings" })
          : translate("zh-CN", en);
      expect(zh).not.toBe(en);
      expect(await screen.findByText(zh)).toBeVisible();
      expect(screen.queryByText("连接测试失败：设置")).not.toBeInTheDocument();
      fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
      expect(await screen.findByText(en)).toBeVisible();
      expect(screen.queryByText(zh)).not.toBeInTheDocument();
      expect(requests.filter((request) => request.method === "POST")).toHaveLength(scenario === "load-error" ? 0 : 1);
      if (scenario === "load-error") {
        expect(requests.filter((request) => request.path === basePath)).toHaveLength(1);
      }
      if (scenario.startsWith("test")) {
        expect(screen.getByLabelText("Host")).toHaveValue("draft.internal");
        expect(requests.filter((request) => request.path === basePath)).toHaveLength(1);
        expect(JSON.parse(String(requests.find((request) => request.method === "POST")?.body))).toMatchObject(
          target === "cache"
            ? { cache_settings: { type: "redis", host: "draft.internal", port: "6379" } }
            : { settings: { host: "draft.internal", port: 6379 } },
        );
      }
    });
  });
});

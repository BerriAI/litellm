import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../../../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import CacheSettings from "./index";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { toast } from "@/lib/toast";
import { translate } from "@/i18n/language";
import { FormProvider, useForm } from "react-hook-form";
import CacheFormField from "./CacheFormField";
import { CACHE_FIELDS } from "./cacheSettingsFields";
import CoordinationRedisFormField from "../coordination_redis_settings/CoordinationRedisFormField";
import { COORDINATION_FIELDS } from "../coordination_redis_settings/coordinationRedisFields";

const { getCacheSettingsCall, testCacheConnectionCall, updateCacheSettingsCall } = vi.hoisted(() => ({
  getCacheSettingsCall: vi.fn(),
  testCacheConnectionCall: vi.fn(),
  updateCacheSettingsCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  getCacheSettingsCall,
  testCacheConnectionCall,
  updateCacheSettingsCall,
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

const LOADED_WITH_ADVANCED = {
  current_values: {
    host: "redis.internal",
    namespace: "prod-ns",
    ttl: "300",
    max_connections: "50",
    ssl: true,
  },
};

const renderSettings = () => renderWithProviders(<CacheSettings accessToken="sk-test" userRole="Admin" userID="u1" />);

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: /save changes/i }));

// 同名服务端文案不能因本地校验错误接入翻译而被改写。
const ServerErrorForm = ({ target }: { target: "cache" | "coordination" }) => {
  const form = useForm({ defaultValues: { port: "99999" } });
  const cachePort = CACHE_FIELDS.find((candidate) => candidate.name === "port")!;
  const coordinationPort = COORDINATION_FIELDS.find((candidate) => candidate.name === "port")!;
  return (
    <FormProvider {...form}>
      {target === "cache" ? (
        <CacheFormField field={cachePort} embeddingModels={[]} />
      ) : (
        <CoordinationRedisFormField field={coordinationPort} isSecretConfigured={false} />
      )}
      <button
        onClick={() =>
          form.setError("port", { type: "server", message: "Port must be an integer between 1 and 65535" })
        }
      >
        返回服务端错误
      </button>
    </FormProvider>
  );
};

describe("CacheSettings advanced settings round-trip", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    getCacheSettingsCall.mockResolvedValue(LOADED_WITH_ADVANCED);
    updateCacheSettingsCall.mockResolvedValue({ status: "success" });
    testCacheConnectionCall.mockResolvedValue({ status: "success" });
  });

  it("中文连接测试错误翻译固定前缀，保留服务器详情", async () => {
    testCacheConnectionCall.mockResolvedValue({ status: "error", message: "Settings" });
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="cache-error" storageScope="test">
        <CacheSettings accessToken="sk-test" userRole="Admin" userID="u1" />
      </LanguageProvider>,
    );
    await screen.findByDisplayValue("redis.internal");
    fireEvent.click(screen.getByRole("button", { name: translate("zh-CN", "Test Connection") }));
    await waitFor(() => expect(toast.fromError).toHaveBeenCalledTimes(1));
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="cache-error" storageScope="test">
        {vi.mocked(toast.fromError).mock.calls[0][1]?.description}
      </LanguageProvider>,
    );
    expect(screen.getByText("连接测试失败：Settings")).toBeVisible();
  });

  // 已出现的本地校验错误也要切换语言，且不能重载或提交配置。
  it("切换语言即时更新端口校验错误并保留无效草稿", async () => {
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="cache-validation" storageScope="test">
        <LanguageSelector />
        <CacheSettings accessToken="sk-test" userRole="Admin" userID="u1" />
      </LanguageProvider>,
    );
    await screen.findByDisplayValue("redis.internal");
    fireEvent.change(screen.getByLabelText("Port"), { target: { value: "99999" } });
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Port must be an integer between 1 and 65535");

    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByRole("alert")).toHaveTextContent("端口必须是 1 到 65535 之间的整数");
    expect(screen.getByLabelText("端口")).toHaveValue("99999");
    expect(screen.getByLabelText("端口")).toHaveAttribute("aria-invalid", "true");
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByRole("alert")).toHaveTextContent("Port must be an integer between 1 and 65535");
    expect(getCacheSettingsCall).toHaveBeenCalledTimes(1);
    expect(updateCacheSettingsCall).not.toHaveBeenCalled();
    expect(testCacheConnectionCall).not.toHaveBeenCalled();
  });

  it.each(["cache", "coordination"] as const)("%s 字段与本地校验同名的服务端错误不翻译", async (target) => {
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="redis-server-error" storageScope="test">
        <LanguageSelector />
        <ServerErrorForm target={target} />
      </LanguageProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "返回服务端错误" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Port must be an integer between 1 and 65535");
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByRole("alert")).toHaveTextContent("Port must be an integer between 1 and 65535");
    expect(screen.getByLabelText("端口")).toHaveValue("99999");
  });

  // 语言变化只更新标签，不能通过加载回调重置 Redis 配置草稿。
  it("切换语言保留未保存的 Redis 地址且不重新加载或写入配置", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="cache-test" storageScope="test">
        <LanguageSelector />
        <CacheSettings accessToken="sk-test" userRole="Admin" userID="u1" />
      </LanguageProvider>,
    );
    expect(await screen.findByRole("textbox", { name: "Host", exact: true })).toHaveValue("redis.internal");
    fireEvent.change(screen.getByRole("textbox", { name: "Host", exact: true }), {
      target: { value: "draft.internal" },
    });
    await user.selectOptions(screen.getByRole("combobox", { name: "Interface language" }), "zh-CN");
    expect(screen.getByRole("textbox", { name: "主机", exact: true })).toHaveValue("draft.internal");
    expect(getCacheSettingsCall).toHaveBeenCalledTimes(1);
    expect(updateCacheSettingsCall).not.toHaveBeenCalled();
    expect(testCacheConnectionCall).not.toHaveBeenCalled();
  });

  it("keeps loaded advanced values in the payload when the section is never opened", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    expect(screen.queryByLabelText("Namespace")).not.toBeInTheDocument();
    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).toEqual({
      type: "redis",
      host: "redis.internal",
      port: "6379",
      ssl: true,
      ssl_check_hostname: false,
      namespace: "prod-ns",
      ttl: 300,
      max_connections: 50,
    });
  });

  it("reveals the advanced field sections only after the user expands them", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");
    expect(screen.queryByText("SSL Settings")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Advanced Settings" }));

    expect(await screen.findByText("SSL Settings")).toBeInTheDocument();
    expect(screen.getByText("Cache Management")).toBeInTheDocument();
    expect(screen.getByText("GCP Authentication")).toBeInTheDocument();
  });

  it("sends the same payload whether or not the advanced section was expanded", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");
    await save(user);
    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    const whileCollapsed = updateCacheSettingsCall.mock.calls[0][1];

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByLabelText("Namespace");
    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(2));
    expect(updateCacheSettingsCall.mock.calls[1][1]).toEqual(whileCollapsed);
  });

  it("preserves a value typed into the advanced section after it is collapsed again", async () => {
    getCacheSettingsCall.mockResolvedValue({ current_values: { host: "redis.internal" } });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    await user.click(screen.getByText("Advanced Settings"));
    fireEvent.change(await screen.findByLabelText("Namespace"), { target: { value: "typed-ns" } });
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => expect(screen.queryByLabelText("Namespace")).not.toBeInTheDocument());

    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).toMatchObject({ namespace: "typed-ns" });
  });

  it("restores a value typed into the advanced section when it is expanded again", async () => {
    getCacheSettingsCall.mockResolvedValue({ current_values: { host: "redis.internal" } });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    await user.click(screen.getByText("Advanced Settings"));
    fireEvent.change(await screen.findByLabelText("Namespace"), { target: { value: "typed-ns" } });
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => expect(screen.queryByLabelText("Namespace")).not.toBeInTheDocument());
    await user.click(screen.getByText("Advanced Settings"));

    expect(await screen.findByLabelText("Namespace")).toHaveValue("typed-ns");
  });

  it("keeps a hidden section's fields out of the payload when the redis type does not use them", async () => {
    getCacheSettingsCall.mockResolvedValue({
      current_values: { redis_type: "sentinel", service_name: "mymaster", host: "redis.internal" },
    });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText("Service Name");
    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).toMatchObject({ service_name: "mymaster" });
    expect(updateCacheSettingsCall.mock.calls[0][1]).not.toHaveProperty("redis_startup_nodes");
  });

  it("saves the semantic cache scope picked from the select and shows the loaded value", async () => {
    getCacheSettingsCall.mockResolvedValue({
      current_values: { redis_type: "semantic", host: "redis.internal", semantic_cache_scope: "key" },
    });
    const user = userEvent.setup();
    renderSettings();
    const trigger = await screen.findByLabelText("Semantic Cache Scope");
    expect(trigger).toHaveTextContent("Key (shared by all end users of the key/team/org)");

    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "End user (isolated per end user)" }));
    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).toMatchObject({
      type: "redis-semantic",
      semantic_cache_scope: "end_user",
    });
  });

  it("does not block the save on a malformed value inside a collapsed advanced section", async () => {
    getCacheSettingsCall.mockResolvedValue({ current_values: { host: "redis.internal" } });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    await user.click(screen.getByText("Advanced Settings"));
    fireEvent.change(await screen.findByLabelText("TTL (seconds)"), { target: { value: "not-a-number" } });
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => expect(screen.queryByLabelText("TTL (seconds)")).not.toBeInTheDocument());

    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).not.toHaveProperty("ttl");
  });

  it("should omit a cleared cache model from save and test while retaining other settings", async () => {
    vi.mocked(fetchAvailableModels).mockResolvedValue([
      { model_group: "synthetic-embedding", mode: "embedding" },
    ] as Awaited<ReturnType<typeof fetchAvailableModels>>);
    getCacheSettingsCall.mockResolvedValue({
      current_values: {
        redis_type: "semantic",
        host: "localhost",
        redis_semantic_cache_embedding_model: "synthetic-embedding",
        password: "***REDACTED***",
        ttl: 0,
        ssl: false,
        namespace: "synthetic-cache",
      },
    });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByRole("combobox", { name: "Embedding Model" });
    await user.click(screen.getByRole("button", { name: "Clear" }));
    const expected = {
      type: "redis",
      host: "localhost",
      port: "6379",
      similarity_threshold: 0.8,
      semantic_cache_scope: "key",
      ssl: false,
      ssl_check_hostname: false,
      ttl: 0,
      namespace: "synthetic-cache",
    };
    await user.click(screen.getByRole("button", { name: "Test Connection" }));
    await waitFor(() => expect(testCacheConnectionCall).toHaveBeenCalledWith("sk-test", expected));
    await save(user);
    await waitFor(() =>
      expect(updateCacheSettingsCall).toHaveBeenCalledWith("sk-test", { ...expected, type: "redis-semantic" }),
    );
  });
});

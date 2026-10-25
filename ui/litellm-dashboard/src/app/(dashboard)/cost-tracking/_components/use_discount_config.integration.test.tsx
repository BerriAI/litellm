import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, screen } from "@testing-library/react";
import { renderCostConfigHook as renderHook, changeCostConfigLanguage } from "../../../../../tests/CostConfigLanguage";
import { useDiscountConfig } from "./use_discount_config";
import { toast } from "@/lib/toast";
import { Providers } from "@/components/provider_info_helpers";

vi.unmock("@/lib/toast");

describe("useDiscountConfig", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    act(() => toast.dismiss());
    vi.restoreAllMocks();
  });

  // 同一个请求完成前后切换语言，既不能重发请求，也不能保留旧语言的通知。
  it.each(["success", "fallback", "save failure", "fetch failure"] as const)(
    "%s 通知在请求等待中和显示后使用当前语言",
    async (outcome) => {
      const request = Promise.withResolvers<Response>();
      const fetchSpy = vi.spyOn(global, "fetch").mockReturnValueOnce(request.promise);
      fetchSpy.mockResolvedValue({ ok: true, json: async () => ({ values: { "openai/Home": 0.05 } }) } as Response);
      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));
      let pending: Promise<void>;
      act(() => {
        pending =
          outcome === "fetch failure"
            ? result.current.fetchDiscountConfig()
            : result.current.saveDiscountConfig({ "openai/Home": 0.05 });
      });
      changeCostConfigLanguage("zh-CN");
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      await act(async () => {
        if (outcome.endsWith("failure")) request.reject(new Error("test-only-internal-error"));
        else request.resolve({ ok: outcome === "success", json: async () => ({ detail: null }) } as Response);
        await pending!;
      });
      const messages = {
        success: ["折扣配置已更新", "Discount configuration updated successfully"],
        fallback: ["设置更新失败", "Failed to update settings"],
        "save failure": ["折扣配置更新失败", "Failed to update discount configuration"],
        "fetch failure": ["折扣配置加载失败", "Failed to fetch discount configuration"],
      } as const;
      expect(await screen.findByText(messages[outcome][0])).toBeInTheDocument();
      changeCostConfigLanguage("en");
      expect(screen.getByText(messages[outcome][1])).toBeInTheDocument();
      expect(screen.queryByText(messages[outcome][0])).not.toBeInTheDocument();
      expect(screen.queryByText("test-only-internal-error")).not.toBeInTheDocument();
      expect(fetchSpy).toHaveBeenCalledTimes(outcome === "success" ? 2 : 1);
      if (outcome !== "fetch failure") {
        expect(fetchSpy.mock.calls[0][1]).toMatchObject({
          method: "PATCH",
          body: JSON.stringify({ "openai/Home": 0.05 }),
        });
      }
    },
  );

  it.each([
    [undefined, "5", "Please select a provider and enter discount percentage", "请选择提供商并输入折扣百分比"],
    ["OpenAI", "101", "Discount must be between 0% and 100%", "折扣必须在 0% 至 100% 之间"],
    ["Unknown", "5", "Invalid provider selected", "所选提供商无效"],
  ] as const)("%s/%s 的已显示校验随语言更新且不保存", async (provider, value, english, chinese) => {
    const fetchSpy = vi.spyOn(global, "fetch");
    const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));
    await act(async () => expect(await result.current.handleAddProvider(provider, value, "Home")).toBe(false));
    expect(await screen.findByText(english)).toBeInTheDocument();
    changeCostConfigLanguage("zh-CN");
    expect(screen.getByText(chinese)).toBeInTheDocument();
    changeCostConfigLanguage("en");
    expect(screen.getByText(english)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it.each(["", "Home"])("重复折扣提示翻译固定范围但保留模型模式 %s", async (pattern) => {
    const key = pattern ? `openai/${pattern}` : "openai";
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ values: { [key]: 0 } }),
    } as Response);
    const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));
    await act(async () => result.current.fetchDiscountConfig());
    await act(async () => expect(await result.current.handleAddProvider("OpenAI", "5", pattern)).toBe(false));
    const english = `Discount for OpenAI (${pattern || "all models"}) already exists. Edit it in the table above.`;
    expect(await screen.findByText(english)).toBeInTheDocument();
    changeCostConfigLanguage("zh-CN");
    expect(
      screen.getByText(`OpenAI（${pattern || "全部模型"}）的折扣已存在，请在上方表格中修改。`),
    ).toBeInTheDocument();
    expect(screen.queryByText(/首页/)).not.toBeInTheDocument();
    changeCostConfigLanguage("en");
    expect(screen.getByText(english)).toBeInTheDocument();
    expect(result.current.discountConfig).toEqual({ [key]: 0 });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it.each(["Home", { error: { message: "Home", debug: "test-only-debug-field" } }])(
    "服务端 403 仅显示原始业务消息并保留权限分类 %j",
    async (detail) => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: async () => ({ detail }),
      } as Response);
      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));
      await act(async () => result.current.saveDiscountConfig({ "openai/Home": 0.05 }));
      expect(await screen.findByText("Home")).toBeInTheDocument();
      expect(screen.getByText("Access Denied")).toBeInTheDocument();
      changeCostConfigLanguage("zh-CN");
      expect(screen.getByText("无访问权限")).toBeInTheDocument();
      expect(screen.getByText("Home")).toBeInTheDocument();
      expect(screen.queryByText(/首页|test-only-debug-field/)).not.toBeInTheDocument();
    },
  );

  describe("fetchDiscountConfig", () => {
    it("should populate discountConfig with fetched values on success", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ values: { openai: 0.05, anthropic: 0.1 } }),
      } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      expect(result.current.discountConfig).toEqual({ openai: 0.05, anthropic: 0.1 });
    });

    it("should set an empty config when the response has no values", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ values: {} }),
      } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      expect(result.current.discountConfig).toEqual({});
    });

    it("should show an error notification when the fetch throws", async () => {
      vi.spyOn(global, "fetch").mockRejectedValueOnce(new Error("Network error"));

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      expect(await screen.findByText("Failed to fetch discount configuration")).toBeInTheDocument();
    });
  });

  describe("handleAddProvider", () => {
    it("should return false and notify when no provider is selected", async () => {
      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider(undefined, "5", "");
      });

      expect(success!).toBe(false);
      expect(await screen.findByText("Please select a provider and enter discount percentage")).toBeInTheDocument();
    });

    it("should return false and notify when no discount is provided", async () => {
      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "", "");
      });

      expect(success!).toBe(false);
      expect(await screen.findByText("Please select a provider and enter discount percentage")).toBeInTheDocument();
    });

    it("should return false and notify when the discount exceeds 100", async () => {
      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "150", "");
      });

      expect(success!).toBe(false);
      expect(await screen.findByText("Discount must be between 0% and 100%")).toBeInTheDocument();
    });

    it("should return false and notify when the provider already exists in the config", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.05 } }),
        } as Response)
        .mockResolvedValue({ ok: true, json: async () => ({}) } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "10", "");
      });

      expect(success!).toBe(false);
      expect(
        await screen.findByText("Discount for OpenAI (all models) already exists. Edit it in the table above."),
      ).toBeInTheDocument();
    });

    it("should reject a duplicate even when the stored discount is 0", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ values: { vertex_ai: 0 } }),
      } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      vi.clearAllMocks();

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("Vertex_AI", "5", "");
      });

      expect(success!).toBe(false);
      expect(
        await screen.findByText(
          `Discount for ${Providers.Vertex_AI} (all models) already exists. Edit it in the table above.`,
        ),
      ).toBeInTheDocument();
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it("should save the config and return true on a valid new provider", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: {} }) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: { openai: 0.05 } }) } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "5", "");
      });

      expect(success!).toBe(true);
      expect(await screen.findByText("Discount configuration updated successfully")).toBeInTheDocument();
    });

    it("should store a <provider>/<pattern> key when a model pattern is given", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: {} }) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: { "openai/gpt-*": 0.2 } }) } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "20", "gpt-*");
      });

      expect(success!).toBe(true);
      const patchCall = vi.mocked(global.fetch).mock.calls.find(([, init]) => init?.method === "PATCH");
      expect(patchCall).toBeDefined();
      expect(JSON.parse(patchCall![1]!.body as string)).toEqual({ "openai/gpt-*": 0.2 });
      expect(result.current.discountConfig).toHaveProperty("openai/gpt-*", 0.2);
    });

    it("should store the bare provider key when the pattern is empty or whitespace", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: { openai: 0.05 } }) } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "5", "   ");
      });

      expect(success!).toBe(true);
      const patchCall = vi.mocked(global.fetch).mock.calls.find(([, init]) => init?.method === "PATCH");
      expect(JSON.parse(patchCall![1]!.body as string)).toEqual({ openai: 0.05 });
    });

    it("should compose a nested pattern with slashes into the config key", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { "openrouter/anthropic/claude-*": 0.15 } }),
        } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("Openrouter", "15", "anthropic/claude-*");
      });

      expect(success!).toBe(true);
      const patchCall = vi.mocked(global.fetch).mock.calls.find(([, init]) => init?.method === "PATCH");
      expect(JSON.parse(patchCall![1]!.body as string)).toEqual({ "openrouter/anthropic/claude-*": 0.15 });
    });

    it("should let a provider with a bare discount also get a pattern entry", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: { openai: 0.05 } }) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.05, "openai/gpt-*": 0.2 } }),
        } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "20", "gpt-*");
      });

      expect(success!).toBe(true);
      const patchCall = vi.mocked(global.fetch).mock.calls.find(([, init]) => init?.method === "PATCH");
      expect(JSON.parse(patchCall![1]!.body as string)).toEqual({ openai: 0.05, "openai/gpt-*": 0.2 });
    });

    it("should reject a duplicate pattern key", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ values: { "openai/gpt-*": 0.2 } }),
      } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddProvider("OpenAI", "10", "gpt-*");
      });

      expect(success!).toBe(false);
      expect(
        await screen.findByText("Discount for OpenAI (gpt-*) already exists. Edit it in the table above."),
      ).toBeInTheDocument();
    });
  });

  describe("handleRemoveProvider", () => {
    it("should remove the provider from the config and save", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.05, anthropic: 0.1 } }),
        } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { anthropic: 0.1 } }),
        } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      expect(result.current.discountConfig).toHaveProperty("openai");

      await act(async () => {
        await result.current.handleRemoveProvider("openai");
      });

      // The optimistic update removes openai immediately
      expect(result.current.discountConfig).not.toHaveProperty("openai");
    });
  });

  describe("handleDiscountChange", () => {
    it("should update the discount value and save when the value is valid", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.05 } }),
        } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.1 } }),
        } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      await act(async () => {
        await result.current.handleDiscountChange("openai", "0.1");
      });

      // Optimistic update applied immediately
      expect(result.current.discountConfig["openai"]).toBe(0.1);
    });

    it("should not save when the value is greater than 1 (invalid fraction)", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ values: { openai: 0.05 } }),
      } as Response);

      const { result } = renderHook(() => useDiscountConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchDiscountConfig();
      });

      // Clear mocks after the initial fetch so we can check that no PATCH was made
      vi.clearAllMocks();

      await act(async () => {
        await result.current.handleDiscountChange("openai", "1.5");
      });

      expect(global.fetch).not.toHaveBeenCalled();
    });
  });
});

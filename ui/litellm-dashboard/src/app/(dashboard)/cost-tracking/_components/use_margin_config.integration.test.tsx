import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, screen } from "@testing-library/react";
import { renderCostConfigHook as renderHook, changeCostConfigLanguage } from "../../../../../tests/CostConfigLanguage";
import { useMarginConfig } from "./use_margin_config";
import { toast } from "@/lib/toast";

vi.unmock("@/lib/toast");

describe("useMarginConfig", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    act(() => toast.dismiss());
    vi.restoreAllMocks();
  });

  // 请求与数值不依赖语言，只有本地通知在显示时翻译。
  it.each(["success", "fallback", "save failure", "fetch failure"] as const)(
    "%s 通知在请求等待中和显示后使用当前语言",
    async (outcome) => {
      const config = { global: { fixed_amount: 0.001 } };
      const request = Promise.withResolvers<Response>();
      const fetchSpy = vi.spyOn(global, "fetch").mockReturnValueOnce(request.promise);
      fetchSpy.mockResolvedValue({ ok: true, json: async () => ({ values: config }) } as Response);
      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      let pending: Promise<void>;
      act(() => {
        pending =
          outcome === "fetch failure" ? result.current.fetchMarginConfig() : result.current.saveMarginConfig(config);
      });
      changeCostConfigLanguage("zh-CN");
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      await act(async () => {
        if (outcome.endsWith("failure")) request.reject(new Error("test-only-internal-error"));
        else request.resolve({ ok: outcome === "success", json: async () => ({ detail: null }) } as Response);
        await pending!;
      });
      const messages = {
        success: ["加价配置已更新", "Margin configuration updated successfully"],
        fallback: ["设置更新失败", "Failed to update settings"],
        "save failure": ["加价配置更新失败", "Failed to update margin configuration"],
        "fetch failure": ["加价配置加载失败", "Failed to fetch margin configuration"],
      } as const;
      expect(await screen.findByText(messages[outcome][0])).toBeInTheDocument();
      changeCostConfigLanguage("en");
      expect(screen.getByText(messages[outcome][1])).toBeInTheDocument();
      expect(screen.queryByText(messages[outcome][0])).not.toBeInTheDocument();
      expect(screen.queryByText("test-only-internal-error")).not.toBeInTheDocument();
      expect(fetchSpy).toHaveBeenCalledTimes(outcome === "success" ? 2 : 1);
      if (outcome !== "fetch failure") {
        expect(fetchSpy.mock.calls[0][1]).toMatchObject({ method: "PATCH", body: JSON.stringify(config) });
      }
    },
  );

  it.each([
    [
      { selectedProvider: undefined, marginType: "percentage", percentageValue: "5", fixedAmountValue: "" },
      "Please select a provider",
      "请选择提供商",
    ],
    [
      { selectedProvider: "Unknown", marginType: "percentage", percentageValue: "5", fixedAmountValue: "" },
      "Invalid provider selected",
      "所选提供商无效",
    ],
    [
      { selectedProvider: "OpenAI", marginType: "percentage", percentageValue: "1001", fixedAmountValue: "" },
      "Percentage must be between 0% and 1000%",
      "百分比必须在 0% 至 1000% 之间",
    ],
    [
      { selectedProvider: "OpenAI", marginType: "fixed", percentageValue: "", fixedAmountValue: "-1" },
      "Fixed amount must be non-negative",
      "固定金额不能为负数",
    ],
  ] as const)("%j 的已显示校验随语言更新且不保存", async (params, english, chinese) => {
    const fetchSpy = vi.spyOn(global, "fetch");
    const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
    await act(async () => {
      expect(await result.current.handleAddMargin(params)).toBe(false);
    });
    expect(await screen.findByText(english)).toBeInTheDocument();
    changeCostConfigLanguage("zh-CN");
    expect(screen.getByText(chinese)).toBeInTheDocument();
    changeCostConfigLanguage("en");
    expect(screen.getByText(english)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it.each(["global", "OpenAI"])("%s 重复加价提示的固定标签随语言更新", async (provider) => {
    const key = provider === "global" ? "global" : "openai";
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ values: { [key]: { fixed_amount: 0.001 } } }),
    } as Response);
    const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
    await act(async () => result.current.fetchMarginConfig());
    const params = {
      selectedProvider: provider,
      marginType: "percentage",
      percentageValue: "5",
      fixedAmountValue: "",
    } as const;
    await act(async () => expect(await result.current.handleAddMargin(params)).toBe(false));
    const english = `Margin for ${provider === "global" ? "Global" : provider} already exists. Edit it in the table above.`;
    expect(await screen.findByText(english)).toBeInTheDocument();
    changeCostConfigLanguage("zh-CN");
    expect(
      screen.getByText(
        provider === "global" ? "全局加价已存在，请在上方表格中修改。" : "OpenAI 的加价已存在，请在上方表格中修改。",
      ),
    ).toBeInTheDocument();
    changeCostConfigLanguage("en");
    expect(screen.getByText(english)).toBeInTheDocument();
    expect(result.current.marginConfig).toEqual({ [key]: { fixed_amount: 0.001 } });
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
      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      await act(async () => result.current.saveMarginConfig({ global: 0.05 }));
      expect(await screen.findByText("Home")).toBeInTheDocument();
      expect(screen.getByText("Access Denied")).toBeInTheDocument();
      changeCostConfigLanguage("zh-CN");
      expect(screen.getByText("无访问权限")).toBeInTheDocument();
      expect(screen.getByText("Home")).toBeInTheDocument();
      expect(screen.queryByText(/首页|test-only-debug-field/)).not.toBeInTheDocument();
    },
  );

  describe("fetchMarginConfig", () => {
    it("should populate marginConfig with fetched values on success", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ values: { openai: 0.1, global: 0.05 } }),
      } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      expect(result.current.marginConfig).toEqual({ openai: 0.1, global: 0.05 });
    });

    it("should set an empty config when the response has no values", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ values: {} }),
      } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      expect(result.current.marginConfig).toEqual({});
    });

    it("should show an error notification when fetch throws", async () => {
      vi.spyOn(global, "fetch").mockRejectedValueOnce(new Error("Network error"));

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      expect(await screen.findByText("Failed to fetch margin configuration")).toBeInTheDocument();
    });
  });

  describe("handleAddMargin", () => {
    it("should return false and notify when no provider is selected", async () => {
      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      const params = {
        selectedProvider: undefined,
        marginType: "percentage",
        percentageValue: "10",
        fixedAmountValue: "",
      } as const;

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin(params);
      });

      expect(success!).toBe(false);
      expect(await screen.findByText("Please select a provider")).toBeInTheDocument();
    });

    it("should return false and notify when percentage is out of range", async () => {
      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      const params = {
        selectedProvider: "OpenAI",
        marginType: "percentage",
        percentageValue: "2000",
        fixedAmountValue: "",
      } as const;

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin(params);
      });

      expect(success!).toBe(false);
      expect(await screen.findByText("Percentage must be between 0% and 1000%")).toBeInTheDocument();
    });

    it("should return false when the provider already has a margin configured", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.05 } }),
        } as Response)
        .mockResolvedValue({ ok: true, json: async () => ({}) } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      const params = {
        selectedProvider: "OpenAI",
        marginType: "percentage",
        percentageValue: "10",
        fixedAmountValue: "",
      } as const;

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin(params);
      });

      expect(success!).toBe(false);
      expect(
        await screen.findByText("Margin for OpenAI already exists. Edit it in the table above."),
      ).toBeInTheDocument();
    });

    it("should save a percentage margin and return true for a valid new provider", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: {} }) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: { openai: 0.1 } }) } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      const params = {
        selectedProvider: "OpenAI",
        marginType: "percentage",
        percentageValue: "10",
        fixedAmountValue: "",
      } as const;

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin(params);
      });

      expect(success!).toBe(true);
      expect(await screen.findByText("Margin configuration updated successfully")).toBeInTheDocument();
    });

    it("should save a fixed amount margin and return true for a valid new provider", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: {} }) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: { fixed_amount: 0.001 } } }),
        } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      const params = {
        selectedProvider: "OpenAI",
        marginType: "fixed",
        percentageValue: "",
        fixedAmountValue: "0.001",
      } as const;

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin(params);
      });

      expect(success!).toBe(true);
      expect(await screen.findByText("Margin configuration updated successfully")).toBeInTheDocument();
    });

    it("should accept the global provider without provider_map lookup", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: {} }) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { global: 0.05 } }),
        } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));
      const params = {
        selectedProvider: "global",
        marginType: "percentage",
        percentageValue: "5",
        fixedAmountValue: "",
      } as const;

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin(params);
      });

      expect(success!).toBe(true);
    });
  });

  describe("handleRemoveMargin", () => {
    it("should remove the provider from the config and save", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.1, anthropic: 0.05 } }),
        } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { anthropic: 0.05 } }),
        } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      expect(result.current.marginConfig).toHaveProperty("openai");

      await act(async () => {
        await result.current.handleRemoveMargin("openai");
      });

      expect(result.current.marginConfig).not.toHaveProperty("openai");
    });
  });

  describe("handleMarginChange", () => {
    it("should update the margin value for a provider and save", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.1 } }),
        } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.2 } }),
        } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      await act(async () => {
        await result.current.handleMarginChange("openai", 0.2);
      });

      // Optimistic update applied immediately
      expect(result.current.marginConfig["openai"]).toBe(0.2);
    });

    it("should update the margin with a complex value (percentage + fixed)", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.1 } }),
        } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            values: { openai: { percentage: 0.05, fixed_amount: 0.001 } },
          }),
        } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      await act(async () => {
        await result.current.handleMarginChange("openai", { percentage: 0.05, fixed_amount: 0.001 });
      });

      expect(result.current.marginConfig["openai"]).toEqual({
        percentage: 0.05,
        fixed_amount: 0.001,
      });
    });
  });
});

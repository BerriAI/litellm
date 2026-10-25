import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, screen } from "@testing-library/react";
import { renderCostConfigHook as renderHook, changeCostConfigLanguage } from "../../../../../tests/CostConfigLanguage";
import { useBlockUnpricedConfig } from "./use_block_unpriced_config";
import { apiClient } from "@/components/networking";
import { toast } from "@/lib/toast";

vi.unmock("@/lib/toast");

vi.mock("@/components/networking", () => ({
  apiClient: {
    get: vi.fn(),
    patch: vi.fn(),
  },
}));

const ENDPOINT = "/config/block_requests_for_models_without_pricing";

describe("useBlockUnpricedConfig", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each([true, false])("未计价模型开关 %s 的成功通知随语言更新且不重发请求", async (enabled) => {
    const request = Promise.withResolvers<{ enabled: boolean }>();
    vi.mocked(apiClient.patch).mockReturnValueOnce(request.promise);
    const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));
    let pending: Promise<void>;
    act(() => {
      pending = result.current.setBlockUnpriced(enabled);
    });
    expect(result.current.isUpdating).toBe(true);
    changeCostConfigLanguage("zh-CN");
    expect(result.current.isUpdating).toBe(true);
    expect(apiClient.patch).toHaveBeenCalledTimes(1);
    await act(async () => {
      request.resolve({ enabled });
      await pending!;
    });
    const chinese = enabled ? "已阻止未定价模型的请求" : "已允许未定价模型的请求";
    const english = enabled
      ? "Requests for models without pricing will now be blocked"
      : "Requests for models without pricing are now allowed";
    expect(await screen.findByText(chinese)).toBeInTheDocument();
    changeCostConfigLanguage("en");
    expect(screen.getByText(english)).toBeInTheDocument();
    expect(screen.queryByText(chinese)).not.toBeInTheDocument();
    expect(result.current.blockUnpriced).toBe(enabled);
    expect(result.current.isUpdating).toBe(false);
    expect(apiClient.get).not.toHaveBeenCalled();
    expect(apiClient.patch).toHaveBeenCalledExactlyOnceWith(ENDPOINT, { accessToken: "test-token", body: { enabled } });
  });

  describe("fetchBlockUnpriced", () => {
    it("reflects the enabled flag returned by the proxy", async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ enabled: true });

      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchBlockUnpriced();
      });

      expect(apiClient.get).toHaveBeenCalledWith(ENDPOINT, { accessToken: "test-token" });
      expect(result.current.blockUnpriced).toBe(true);
    });

    it("surfaces a toast when the fetch throws", async () => {
      const error = new Error("Network error");
      vi.mocked(apiClient.get).mockRejectedValueOnce(error);

      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchBlockUnpriced();
      });

      expect(await screen.findByText("Network error")).toBeInTheDocument();
      expect(result.current.blockUnpriced).toBe(false);
    });

    it("does nothing without an access token", async () => {
      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: null }));

      await act(async () => {
        await result.current.fetchBlockUnpriced();
      });

      expect(apiClient.get).not.toHaveBeenCalled();
    });
  });

  describe("setBlockUnpriced", () => {
    it("persists the new value and confirms it with a toast", async () => {
      vi.mocked(apiClient.patch).mockResolvedValueOnce({ enabled: true });

      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.setBlockUnpriced(true);
      });

      expect(apiClient.patch).toHaveBeenCalledWith(ENDPOINT, {
        accessToken: "test-token",
        body: { enabled: true },
      });
      expect(result.current.blockUnpriced).toBe(true);
      expect(await screen.findByText("Requests for models without pricing will now be blocked")).toBeInTheDocument();
      expect(result.current.isUpdating).toBe(false);
    });

    it("confirms turning the block back off", async () => {
      vi.mocked(apiClient.patch).mockResolvedValueOnce({ enabled: false });

      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.setBlockUnpriced(false);
      });

      expect(result.current.blockUnpriced).toBe(false);
      expect(await screen.findByText("Requests for models without pricing are now allowed")).toBeInTheDocument();
    });

    it("surfaces the proxy error and leaves the flag unchanged when the update fails", async () => {
      const error = new Error("Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature.");
      vi.mocked(apiClient.patch).mockRejectedValueOnce(error);

      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.setBlockUnpriced(true);
      });

      expect(await screen.findByText(error.message)).toBeInTheDocument();
      expect(screen.queryByText("Requests for models without pricing will now be blocked")).not.toBeInTheDocument();
      expect(result.current.blockUnpriced).toBe(false);
      expect(result.current.isUpdating).toBe(false);
    });
  });
});

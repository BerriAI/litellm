import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useBlockUnpricedConfig } from "./use_block_unpriced_config";
import { apiClient } from "@/components/networking";
import { toast } from "@/lib/toast";

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

      expect(toast.fromError).toHaveBeenCalledWith(error);
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
      expect(toast.success).toHaveBeenCalledWith(expect.stringMatching(/will now be blocked/i));
      expect(result.current.isUpdating).toBe(false);
    });

    it("confirms turning the block back off", async () => {
      vi.mocked(apiClient.patch).mockResolvedValueOnce({ enabled: false });

      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.setBlockUnpriced(false);
      });

      expect(result.current.blockUnpriced).toBe(false);
      expect(toast.success).toHaveBeenCalledWith(expect.stringMatching(/now allowed/i));
    });

    it("surfaces the proxy error and leaves the flag unchanged when the update fails", async () => {
      const error = new Error("Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature.");
      vi.mocked(apiClient.patch).mockRejectedValueOnce(error);

      const { result } = renderHook(() => useBlockUnpricedConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.setBlockUnpriced(true);
      });

      expect(toast.fromError).toHaveBeenCalledWith(error);
      expect(toast.success).not.toHaveBeenCalled();
      expect(result.current.blockUnpriced).toBe(false);
      expect(result.current.isUpdating).toBe(false);
    });
  });
});

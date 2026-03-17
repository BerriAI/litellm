import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useMarginConfig } from "./use_margin_config";
import NotificationsManager from "@/components/molecules/notifications_manager";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => ""),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

vi.mock("./provider_display_helpers", () => ({
  getProviderBackendValue: vi.fn((enumKey: string) => {
    const map: Record<string, string> = {
      OpenAI: "openai",
      Anthropic: "anthropic",
    };
    return map[enumKey] ?? null;
  }),
}));

vi.mock("../provider_info_helpers", () => ({
  Providers: {
    OpenAI: "OpenAI",
    Anthropic: "Anthropic",
  },
}));

describe("useMarginConfig", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

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

      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
        expect.stringMatching(/failed to fetch/i)
      );
    });
  });

  describe("handleAddMargin", () => {
    it("should return false and notify when no provider is selected", async () => {
      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin({
          selectedProvider: undefined,
          marginType: "percentage",
          percentageValue: "10",
          fixedAmountValue: "",
        });
      });

      expect(success!).toBe(false);
      expect(NotificationsManager.fromBackend).toHaveBeenCalled();
    });

    it("should return false and notify when percentage is out of range", async () => {
      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin({
          selectedProvider: "OpenAI",
          marginType: "percentage",
          percentageValue: "2000",
          fixedAmountValue: "",
        });
      });

      expect(success!).toBe(false);
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
        expect.stringMatching(/0%.*1000%/i)
      );
    });

    it("should return false when the provider already has a margin configured", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ values: { openai: 0.05 } }),
        } as Response)
        .mockResolvedValue({ ok: true, json: async () => ({}) } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin({
          selectedProvider: "OpenAI",
          marginType: "percentage",
          percentageValue: "10",
          fixedAmountValue: "",
        });
      });

      expect(success!).toBe(false);
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
        expect.stringMatching(/already exists/i)
      );
    });

    it("should save a percentage margin and return true for a valid new provider", async () => {
      vi.spyOn(global, "fetch")
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: {} }) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)
        .mockResolvedValueOnce({ ok: true, json: async () => ({ values: { openai: 0.1 } }) } as Response);

      const { result } = renderHook(() => useMarginConfig({ accessToken: "test-token" }));

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin({
          selectedProvider: "OpenAI",
          marginType: "percentage",
          percentageValue: "10",
          fixedAmountValue: "",
        });
      });

      expect(success!).toBe(true);
      expect(NotificationsManager.success).toHaveBeenCalled();
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

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin({
          selectedProvider: "OpenAI",
          marginType: "fixed",
          percentageValue: "",
          fixedAmountValue: "0.001",
        });
      });

      expect(success!).toBe(true);
      expect(NotificationsManager.success).toHaveBeenCalled();
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

      await act(async () => {
        await result.current.fetchMarginConfig();
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.handleAddMargin({
          selectedProvider: "global",
          marginType: "percentage",
          percentageValue: "5",
          fixedAmountValue: "",
        });
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

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useMultiCostEstimate } from "./use_multi_cost_estimate";
import type { ModelEntry } from "./types";
import type { CostEstimateResponse } from "../types";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => ""),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

function makeEntry(overrides: Partial<ModelEntry> = {}): ModelEntry {
  return {
    id: "entry-1",
    model: "gpt-4",
    input_tokens: 1000,
    output_tokens: 500,
    ...overrides,
  };
}

function makeApiResponse(overrides: Partial<CostEstimateResponse> = {}): CostEstimateResponse {
  return {
    model: "gpt-4",
    input_tokens: 1000,
    output_tokens: 500,
    num_requests_per_day: null,
    num_requests_per_month: null,
    cost_per_request: 0.05,
    input_cost_per_request: 0.03,
    output_cost_per_request: 0.02,
    margin_cost_per_request: 0,
    daily_cost: null,
    daily_input_cost: null,
    daily_output_cost: null,
    daily_margin_cost: null,
    monthly_cost: null,
    monthly_input_cost: null,
    monthly_output_cost: null,
    monthly_margin_cost: null,
    input_cost_per_token: 0.00003,
    output_cost_per_token: 0.00004,
    provider: "openai",
    ...overrides,
  };
}

describe("useMultiCostEstimate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("debouncedFetchForEntry", () => {
    it("should not fetch when access token is null", async () => {
      const fetchSpy = vi.spyOn(global, "fetch");
      const { result } = renderHook(() => useMultiCostEstimate(null));

      await act(async () => {
        result.current.debouncedFetchForEntry(makeEntry());
        await vi.runAllTimersAsync();
      });

      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("should not fetch when the model field is empty", async () => {
      const fetchSpy = vi.spyOn(global, "fetch");
      const { result } = renderHook(() => useMultiCostEstimate("token123"));

      await act(async () => {
        result.current.debouncedFetchForEntry(makeEntry({ model: "" }));
        await vi.runAllTimersAsync();
      });

      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("should not fetch immediately — only after the debounce delay", async () => {
      const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue({
        ok: true,
        json: async () => makeApiResponse(),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));

      act(() => {
        result.current.debouncedFetchForEntry(makeEntry());
      });

      expect(fetchSpy).not.toHaveBeenCalled();

      await act(async () => {
        await vi.runAllTimersAsync();
      });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
    });

    it("should cancel an in-flight debounce when called again for the same entry", async () => {
      const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue({
        ok: true,
        json: async () => makeApiResponse(),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));

      await act(async () => {
        result.current.debouncedFetchForEntry(makeEntry());
        vi.advanceTimersByTime(200);
        result.current.debouncedFetchForEntry(makeEntry());
        vi.advanceTimersByTime(200);
        result.current.debouncedFetchForEntry(makeEntry());
        await vi.runAllTimersAsync();
      });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
    });

    it("should store the API result after a successful fetch", async () => {
      vi.spyOn(global, "fetch").mockResolvedValue({
        ok: true,
        json: async () => makeApiResponse(),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const entry = makeEntry();

      await act(async () => {
        result.current.debouncedFetchForEntry(entry);
        await vi.runAllTimersAsync();
      });

      const multiResult = result.current.getMultiModelResult([entry]);
      expect(multiResult.entries[0].result).not.toBeNull();
      expect(multiResult.entries[0].result?.cost_per_request).toBe(0.05);
      expect(multiResult.entries[0].loading).toBe(false);
      expect(multiResult.entries[0].error).toBeNull();
    });

    it("should set an error message when the API returns a non-ok response", async () => {
      vi.spyOn(global, "fetch").mockResolvedValue({
        ok: false,
        json: async () => ({ detail: { error: "Model not found" } }),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const entry = makeEntry();

      await act(async () => {
        result.current.debouncedFetchForEntry(entry);
        await vi.runAllTimersAsync();
      });

      const multiResult = result.current.getMultiModelResult([entry]);
      expect(multiResult.entries[0].result).toBeNull();
      expect(multiResult.entries[0].error).toBe("Model not found");
    });

    it("should fall back to detail string when error has no nested error field", async () => {
      vi.spyOn(global, "fetch").mockResolvedValue({
        ok: false,
        json: async () => ({ detail: "Bad request" }),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const entry = makeEntry();

      await act(async () => {
        result.current.debouncedFetchForEntry(entry);
        await vi.runAllTimersAsync();
      });

      const multiResult = result.current.getMultiModelResult([entry]);
      expect(multiResult.entries[0].error).toBe("Bad request");
    });

    it("should set 'Network error' when fetch throws", async () => {
      vi.spyOn(global, "fetch").mockRejectedValue(new Error("connection refused"));

      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const entry = makeEntry();

      await act(async () => {
        result.current.debouncedFetchForEntry(entry);
        await vi.runAllTimersAsync();
      });

      const multiResult = result.current.getMultiModelResult([entry]);
      expect(multiResult.entries[0].error).toBe("Network error");
      expect(multiResult.entries[0].result).toBeNull();
    });
  });

  describe("removeEntry", () => {
    it("should remove an entry's cached result", async () => {
      vi.spyOn(global, "fetch").mockResolvedValue({
        ok: true,
        json: async () => makeApiResponse(),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const entry = makeEntry();

      await act(async () => {
        result.current.debouncedFetchForEntry(entry);
        await vi.runAllTimersAsync();
      });

      // Confirm result was stored
      expect(result.current.getMultiModelResult([entry]).entries[0].result).not.toBeNull();

      act(() => {
        result.current.removeEntry(entry.id);
      });

      // After removal, the entry should return as if it never fetched
      const multiResult = result.current.getMultiModelResult([entry]);
      expect(multiResult.entries[0].result).toBeNull();
    });

    it("should cancel a pending debounce for the removed entry", async () => {
      const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue({
        ok: true,
        json: async () => makeApiResponse(),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const entry = makeEntry();

      act(() => {
        result.current.debouncedFetchForEntry(entry);
        result.current.removeEntry(entry.id);
      });

      await act(async () => {
        await vi.runAllTimersAsync();
      });

      expect(fetchSpy).not.toHaveBeenCalled();
    });
  });

  describe("getMultiModelResult", () => {
    it("should return zero totals when no entries have results", () => {
      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const multiResult = result.current.getMultiModelResult([makeEntry()]);

      expect(multiResult.totals.cost_per_request).toBe(0);
      expect(multiResult.totals.margin_per_request).toBe(0);
      expect(multiResult.totals.daily_cost).toBeNull();
      expect(multiResult.totals.monthly_cost).toBeNull();
    });

    it("should return an empty entries array for an empty input list", () => {
      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const multiResult = result.current.getMultiModelResult([]);

      expect(multiResult.entries).toHaveLength(0);
      expect(multiResult.totals.daily_cost).toBeNull();
      expect(multiResult.totals.monthly_cost).toBeNull();
    });

    it("should sum cost_per_request across multiple loaded entries", async () => {
      const entry1 = makeEntry({ id: "e1", model: "gpt-4" });
      const entry2 = makeEntry({ id: "e2", model: "claude-3" });

      let callIndex = 0;
      const responses = [
        makeApiResponse({ cost_per_request: 0.05, margin_cost_per_request: 0 }),
        makeApiResponse({ model: "claude-3", cost_per_request: 0.10, margin_cost_per_request: 0 }),
      ];

      vi.spyOn(global, "fetch").mockImplementation(async () => ({
        ok: true,
        json: async () => responses[callIndex++],
      } as Response));

      const { result } = renderHook(() => useMultiCostEstimate("token123"));

      await act(async () => {
        result.current.debouncedFetchForEntry(entry1);
        result.current.debouncedFetchForEntry(entry2);
        await vi.runAllTimersAsync();
      });

      const multiResult = result.current.getMultiModelResult([entry1, entry2]);
      expect(multiResult.totals.cost_per_request).toBeCloseTo(0.15);
    });

    it("should accumulate daily cost only when entries have a daily cost", async () => {
      const entry1 = makeEntry({ id: "e1", model: "gpt-4" });
      const entry2 = makeEntry({ id: "e2", model: "claude-3" });

      let callIndex = 0;
      const responses = [
        makeApiResponse({ daily_cost: 5.0, daily_margin_cost: 0, monthly_cost: null, monthly_margin_cost: null }),
        makeApiResponse({ model: "claude-3", daily_cost: 10.0, daily_margin_cost: 0, monthly_cost: null, monthly_margin_cost: null }),
      ];

      vi.spyOn(global, "fetch").mockImplementation(async () => ({
        ok: true,
        json: async () => responses[callIndex++],
      } as Response));

      const { result } = renderHook(() => useMultiCostEstimate("token123"));

      await act(async () => {
        result.current.debouncedFetchForEntry(entry1);
        result.current.debouncedFetchForEntry(entry2);
        await vi.runAllTimersAsync();
      });

      const multiResult = result.current.getMultiModelResult([entry1, entry2]);
      expect(multiResult.totals.daily_cost).toBeCloseTo(15.0);
      expect(multiResult.totals.monthly_cost).toBeNull();
    });

    it("should mark each entry's loading and error state from cached data", async () => {
      vi.spyOn(global, "fetch").mockResolvedValue({
        ok: false,
        json: async () => ({ detail: "Not found" }),
      } as Response);

      const { result } = renderHook(() => useMultiCostEstimate("token123"));
      const entry = makeEntry();

      await act(async () => {
        result.current.debouncedFetchForEntry(entry);
        await vi.runAllTimersAsync();
      });

      const multiResult = result.current.getMultiModelResult([entry]);
      expect(multiResult.entries[0].error).toBe("Not found");
      expect(multiResult.entries[0].loading).toBe(false);
    });
  });
});

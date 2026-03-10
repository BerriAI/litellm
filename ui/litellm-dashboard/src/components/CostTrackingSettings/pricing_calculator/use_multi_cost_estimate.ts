import { useState, useCallback, useRef, useEffect } from "react";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { CostEstimateRequest, CostEstimateResponse } from "../types";
import { ModelEntry, MultiModelResult } from "./types";

const DEBOUNCE_MS = 500;

interface EntryResult {
  entry: ModelEntry;
  result: CostEstimateResponse | null;
  loading: boolean;
  error: string | null;
}

export function useMultiCostEstimate(accessToken: string | null) {
  const [entryResults, setEntryResults] = useState<Map<string, EntryResult>>(new Map());
  const debounceRefs = useRef<Map<string, NodeJS.Timeout>>(new Map());

  const fetchEstimateForEntry = useCallback(
    async (entry: ModelEntry) => {
      if (!accessToken || !entry.model) {
        setEntryResults((prev) => {
          const next = new Map(prev);
          next.set(entry.id, {
            entry,
            result: null,
            loading: false,
            error: null,
          });
          return next;
        });
        return;
      }

      setEntryResults((prev) => {
        const next = new Map(prev);
        const existing = next.get(entry.id);
        next.set(entry.id, {
          entry,
          result: existing?.result ?? null,
          loading: true,
          error: null,
        });
        return next;
      });

      try {
        const proxyBaseUrl = getProxyBaseUrl();
        const url = proxyBaseUrl ? `${proxyBaseUrl}/cost/estimate` : "/cost/estimate";

        const requestBody: CostEstimateRequest = {
          model: entry.model,
          input_tokens: entry.input_tokens || 0,
          output_tokens: entry.output_tokens || 0,
          num_requests_per_day: entry.num_requests_per_day || null,
          num_requests_per_month: entry.num_requests_per_month || null,
        };

        const response = await fetch(url, {
          method: "POST",
          headers: {
            [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (response.ok) {
          const data: CostEstimateResponse = await response.json();
          setEntryResults((prev) => {
            const next = new Map(prev);
            next.set(entry.id, {
              entry,
              result: data,
              loading: false,
              error: null,
            });
            return next;
          });
        } else {
          const errorData = await response.json();
          const errorMessage =
            errorData.detail?.error || errorData.detail || "Failed to estimate cost";
          setEntryResults((prev) => {
            const next = new Map(prev);
            next.set(entry.id, {
              entry,
              result: null,
              loading: false,
              error: errorMessage,
            });
            return next;
          });
        }
      } catch (error) {
        console.error("Error estimating cost:", error);
        setEntryResults((prev) => {
          const next = new Map(prev);
          next.set(entry.id, {
            entry,
            result: null,
            loading: false,
            error: "Network error",
          });
          return next;
        });
      }
    },
    [accessToken]
  );

  const debouncedFetchForEntry = useCallback(
    (entry: ModelEntry) => {
      const existingTimeout = debounceRefs.current.get(entry.id);
      if (existingTimeout) {
        clearTimeout(existingTimeout);
      }
      const timeout = setTimeout(() => {
        fetchEstimateForEntry(entry);
      }, DEBOUNCE_MS);
      debounceRefs.current.set(entry.id, timeout);
    },
    [fetchEstimateForEntry]
  );

  const removeEntry = useCallback((id: string) => {
    const timeout = debounceRefs.current.get(id);
    if (timeout) {
      clearTimeout(timeout);
      debounceRefs.current.delete(id);
    }
    setEntryResults((prev) => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  useEffect(() => {
    const refs = debounceRefs.current;
    return () => {
      refs.forEach((timeout) => clearTimeout(timeout));
      refs.clear();
    };
  }, []);

  const getMultiModelResult = useCallback(
    (entries: ModelEntry[]): MultiModelResult => {
      const results: MultiModelResult["entries"] = entries.map((entry) => {
        const cached = entryResults.get(entry.id);
        return {
          entry,
          result: cached?.result ?? null,
          loading: cached?.loading ?? false,
          error: cached?.error ?? null,
        };
      });

      let totalCostPerRequest = 0;
      let totalDailyCost: number | null = null;
      let totalMonthlyCost: number | null = null;
      let totalMarginPerRequest = 0;
      let totalDailyMargin: number | null = null;
      let totalMonthlyMargin: number | null = null;

      for (const r of results) {
        if (r.result) {
          totalCostPerRequest += r.result.cost_per_request;
          totalMarginPerRequest += r.result.margin_cost_per_request;
          if (r.result.daily_cost !== null) {
            totalDailyCost = (totalDailyCost ?? 0) + r.result.daily_cost;
          }
          if (r.result.daily_margin_cost !== null) {
            totalDailyMargin = (totalDailyMargin ?? 0) + r.result.daily_margin_cost;
          }
          if (r.result.monthly_cost !== null) {
            totalMonthlyCost = (totalMonthlyCost ?? 0) + r.result.monthly_cost;
          }
          if (r.result.monthly_margin_cost !== null) {
            totalMonthlyMargin = (totalMonthlyMargin ?? 0) + r.result.monthly_margin_cost;
          }
        }
      }

      return {
        entries: results,
        totals: {
          cost_per_request: totalCostPerRequest,
          daily_cost: totalDailyCost,
          monthly_cost: totalMonthlyCost,
          margin_per_request: totalMarginPerRequest,
          daily_margin: totalDailyMargin,
          monthly_margin: totalMonthlyMargin,
        },
      };
    },
    [entryResults]
  );

  return {
    debouncedFetchForEntry,
    removeEntry,
    getMultiModelResult,
  };
}


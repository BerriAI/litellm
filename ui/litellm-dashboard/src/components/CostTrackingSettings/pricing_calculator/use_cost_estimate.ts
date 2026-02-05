import { useState, useCallback, useRef, useEffect } from "react";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import NotificationsManager from "../../molecules/notifications_manager";
import { CostEstimateRequest, CostEstimateResponse } from "../types";
import { PricingFormValues } from "./types";

const DEBOUNCE_MS = 500;

export function useCostEstimate(accessToken: string | null) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CostEstimateResponse | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const fetchEstimate = useCallback(
    async (values: PricingFormValues) => {
      if (!accessToken || !values.model) {
        setResult(null);
        return;
      }

      setLoading(true);
      try {
        const proxyBaseUrl = getProxyBaseUrl();
        const url = proxyBaseUrl
          ? `${proxyBaseUrl}/cost/estimate`
          : "/cost/estimate";

        const requestBody: CostEstimateRequest = {
          model: values.model,
          input_tokens: values.input_tokens || 0,
          output_tokens: values.output_tokens || 0,
          num_requests_per_day: values.num_requests_per_day || null,
          num_requests_per_month: values.num_requests_per_month || null,
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
          setResult(data);
        } else {
          const errorData = await response.json();
          const errorMessage =
            errorData.detail?.error || errorData.detail || "Failed to estimate cost";
          NotificationsManager.fromBackend(errorMessage);
          setResult(null);
        }
      } catch (error) {
        console.error("Error estimating cost:", error);
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [accessToken]
  );

  const debouncedFetch = useCallback(
    (values: PricingFormValues) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        fetchEstimate(values);
      }, DEBOUNCE_MS);
    },
    [fetchEstimate]
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  return { loading, result, debouncedFetch };
}


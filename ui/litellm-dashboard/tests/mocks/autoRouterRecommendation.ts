import { vi } from "vitest";
import type { AutoRouterRecommendationResponse } from "@/components/networking";

interface AutoRouterRecommendationQuery {
  data: AutoRouterRecommendationResponse | undefined;
  isPending: boolean;
  isError: boolean;
  error: unknown;
  refetch: ReturnType<typeof vi.fn>;
}

export const EMPTY_RECOMMENDATION_QUERY: AutoRouterRecommendationQuery = {
  data: undefined,
  isPending: false,
  isError: false,
  error: null,
  refetch: vi.fn(),
};

export const useAutoRouterRecommendation = vi.fn((): AutoRouterRecommendationQuery => EMPTY_RECOMMENDATION_QUERY);

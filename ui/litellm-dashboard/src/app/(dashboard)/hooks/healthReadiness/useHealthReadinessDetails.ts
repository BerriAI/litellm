import { useQuery, UseQueryResult } from "@tanstack/react-query";
import {
  getGlobalLitellmHeaderName,
  getProxyBaseUrl,
} from "@/components/networking";
import { createQueryKeys } from "../common/queryKeysFactory";

const healthReadinessDetailsKeys = createQueryKeys("healthReadinessDetails");

export interface HealthReadinessDetailsResponse {
  status: string;
  db?: string;
  cache?: unknown;
  litellm_version?: string;
  success_callbacks?: string[];
  use_aiohttp_transport?: boolean;
  log_level?: string;
  is_detailed_debug?: boolean;
}

const fetchHealthReadinessDetails = async (
  accessToken: string,
): Promise<HealthReadinessDetailsResponse> => {
  const baseUrl = getProxyBaseUrl();
  const response = await fetch(`${baseUrl}/health/readiness/details`, {
    method: "GET",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch health readiness details: ${response.statusText}`,
    );
  }
  return response.json();
};

/**
 * Fetches the auth-gated detailed readiness payload.
 *
 * The caller passes its own `accessToken` so this hook stays usable in both
 * authed and unauthed shells (e.g. the public model hub renders the navbar
 * with a null token). When `accessToken` is falsy the query stays disabled
 * and `data` is undefined — consumers should treat that as "details
 * unavailable" rather than an error.
 */
export const useHealthReadinessDetails = (
  accessToken: string | null | undefined,
): UseQueryResult<HealthReadinessDetailsResponse> => {
  return useQuery<HealthReadinessDetailsResponse>({
    queryKey: healthReadinessDetailsKeys.detail("readiness"),
    queryFn: () => fetchHealthReadinessDetails(accessToken!),
    enabled: Boolean(accessToken),
    staleTime: 5 * 60 * 1000,
  });
};

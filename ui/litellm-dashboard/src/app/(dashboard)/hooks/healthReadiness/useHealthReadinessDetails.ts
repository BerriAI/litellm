import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
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

const fetchHealthReadinessDetails = async (accessToken: string): Promise<HealthReadinessDetailsResponse> => {
  const baseUrl = getProxyBaseUrl();
  const response = await fetch(`${baseUrl}/health/readiness/details`, {
    method: "GET",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch health readiness details: ${response.statusText}`);
  }
  return response.json();
};

/**
 * Fetches the admin-only detailed readiness payload.
 *
 * The caller passes its own `accessToken` and `userRole` so this hook stays
 * usable in both authed and unauthed shells (e.g. the public model hub renders
 * the navbar with a null token and no role). When either is missing, or the
 * role cannot read proxy diagnostics, the query stays disabled and `data` is
 * undefined — consumers should treat that as "details unavailable" rather than
 * an error.
 */
export const useHealthReadinessDetails = (
  accessToken: string | null | undefined,
  userRole: string | null | undefined,
): UseQueryResult<HealthReadinessDetailsResponse> => {
  return useQuery<HealthReadinessDetailsResponse>({
    queryKey: healthReadinessDetailsKeys.detail("readiness"),
    queryFn: () => fetchHealthReadinessDetails(accessToken!),
    enabled: Boolean(accessToken) && hasCapability(userRole, "viewProxyDiagnostics"),
    staleTime: 5 * 60 * 1000,
    // The response feeds a passive navbar tag and a debug banner — a failed
    // call (e.g. expired token → 401) shouldn't fan out into three retries.
    retry: false,
  });
};

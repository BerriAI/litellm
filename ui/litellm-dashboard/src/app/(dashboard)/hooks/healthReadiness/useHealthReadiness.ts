import { getProxyBaseUrl } from "@/components/networking";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const healthReadinessKeys = createQueryKeys("healthReadiness");

interface HealthReadinessResponse {
  litellm_version?: string;
  [key: string]: any;
}

const fetchHealthReadiness = async (): Promise<HealthReadinessResponse> => {
  const baseUrl = getProxyBaseUrl();
  const response = await fetch(`${baseUrl}/health/readiness`);
  if (!response.ok) {
    throw new Error(`Failed to fetch health readiness: ${response.statusText}`);
  }
  return response.json();
};

export const useHealthReadiness = (): UseQueryResult<HealthReadinessResponse> => {
  return useQuery<HealthReadinessResponse>({
    queryKey: healthReadinessKeys.detail("readiness"),
    queryFn: fetchHealthReadiness,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

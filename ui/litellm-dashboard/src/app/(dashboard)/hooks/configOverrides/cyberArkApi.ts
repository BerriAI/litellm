import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { createApiClient } from "@/lib/http/client";

export interface CyberArkFieldSchema {
  description?: string;
  properties: Record<string, { description?: string; type?: string }>;
}

export interface CyberArkConfigResponse {
  config_type: string;
  values: Record<string, string | null>;
  field_schema: CyberArkFieldSchema;
}

export interface CyberArkStatusResponse {
  status: string;
  message: string;
}

const apiClient = createApiClient({
  getBaseUrl: getProxyBaseUrl,
  getAuthHeaderName: getGlobalLitellmHeaderName,
});

export const getCyberArkConfig = async (accessToken: string): Promise<CyberArkConfigResponse> =>
  apiClient.get<CyberArkConfigResponse>("/config_overrides/cyberark", { accessToken });

export const updateCyberArkConfig = async (
  accessToken: string,
  config: Record<string, string>,
): Promise<CyberArkStatusResponse> =>
  apiClient.post<CyberArkStatusResponse>("/config_overrides/cyberark", { accessToken, body: config });

export const deleteCyberArkConfig = async (accessToken: string): Promise<CyberArkStatusResponse> =>
  apiClient.delete<CyberArkStatusResponse>("/config_overrides/cyberark", { accessToken });

export const testCyberArkConnection = async (accessToken: string): Promise<CyberArkStatusResponse> =>
  apiClient.post<CyberArkStatusResponse>("/config_overrides/cyberark/test_connection", { accessToken });

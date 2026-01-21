import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { proxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";

export interface RouterSettingsField {
  field_name: string;
  field_type: string;
  field_description: string;
  field_default: any;
  options: string[] | null;
  ui_field_name: string;
  link: string | null;
}

export interface RouterFieldsResponse {
  fields: RouterSettingsField[];
  routing_strategy_descriptions: Record<string, string>;
}

const routerFieldsKeys = createQueryKeys("routerFields");

const deriveErrorMessage = (errorData: any): string => {
  return (
    (errorData?.error && (errorData.error.message || errorData.error)) ||
    errorData?.message ||
    errorData?.detail ||
    errorData?.error ||
    JSON.stringify(errorData)
  );
};

const getRouterFields = async (accessToken: string): Promise<RouterFieldsResponse> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/router/fields` : `/router/fields`;

    console.log("Fetching router fields from:", url);

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = deriveErrorMessage(errorData);
      throw new Error(errorMessage);
    }

    const data: RouterFieldsResponse = await response.json();
    console.log("Fetched router fields:", data);
    return data;
  } catch (error) {
    console.error("Failed to fetch router fields:", error);
    throw error;
  }
};

export const useRouterFields = (): UseQueryResult<RouterFieldsResponse> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<RouterFieldsResponse>({
    queryKey: routerFieldsKeys.detail("fields"),
    queryFn: async () => await getRouterFields(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
  });
};

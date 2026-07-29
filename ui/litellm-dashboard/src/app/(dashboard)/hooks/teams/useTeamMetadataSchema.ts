import { useQuery } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { createApiClient } from "@/lib/http/client";

export interface TeamMetadataField {
  key: string;
  label?: string | null;
  required?: boolean;
  description?: string | null;
}

const TWENTY_FOUR_HOURS_MS = 24 * 60 * 60 * 1000;

const silentApiClient = createApiClient({
  getBaseUrl: getProxyBaseUrl,
  getAuthHeaderName: getGlobalLitellmHeaderName,
});

export const fetchTeamMetadataSchema = async (accessToken: string): Promise<TeamMetadataField[]> => {
  const data = await silentApiClient.get<{ fields?: TeamMetadataField[] }>("/team/metadata_schema", { accessToken });
  return Array.isArray(data?.fields) ? data.fields : [];
};

export const teamMetadataSchemaKeys = createQueryKeys("teamMetadataSchema");

export const useTeamMetadataSchema = () => {
  const { accessToken } = useAuthorized();

  return useQuery<TeamMetadataField[]>({
    queryKey: teamMetadataSchemaKeys.list({}),
    queryFn: async () => await fetchTeamMetadataSchema(accessToken!),
    enabled: Boolean(accessToken),
    staleTime: TWENTY_FOUR_HOURS_MS,
    gcTime: TWENTY_FOUR_HOURS_MS,
    retry: 1,
  });
};

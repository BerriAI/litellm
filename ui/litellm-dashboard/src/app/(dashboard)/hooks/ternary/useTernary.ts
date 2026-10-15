import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";

import { createQueryKeys } from "../common/queryKeysFactory";

export type TernarySettingsView = components["schemas"]["TernarySettingsView"];
export type TernarySettingsInput = components["schemas"]["TernarySettings"];
export type TernarySettingsUpdate = components["schemas"]["TernarySettingsUpdate"];
export type TernaryMessageResponse = components["schemas"]["TernaryMessageResponse"];
export type TernaryDryRunResponse = components["schemas"]["TernaryDryRunResponse"];
export type TernaryExportRequest = components["schemas"]["TernaryExportRequest"];
export type TernaryExportResponse = components["schemas"]["TernaryExportResponse"];

export const ternaryKeys = createQueryKeys("ternarySettings");

const requireToken = (accessToken: string | null | undefined): string => {
  if (!accessToken) {
    throw new Error("Access token is required");
  }
  return accessToken;
};

export const fetchTernarySettings = async (accessToken: string): Promise<TernarySettingsView | null> => {
  const view = await apiClient.get<TernarySettingsView>("/ternary/settings", { accessToken });
  return view.status ? view : null;
};

export const useTernarySettings = () => {
  const { accessToken } = useAuthorized();
  return useQuery({
    queryKey: ternaryKeys.detail("current"),
    queryFn: () => fetchTernarySettings(requireToken(accessToken)),
    enabled: Boolean(accessToken),
  });
};

const useSettingsMutation = <TInput>(
  send: (accessToken: string, input: TInput) => Promise<TernaryMessageResponse>,
) => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  return useMutation<TernaryMessageResponse, Error, TInput>({
    mutationFn: (input) => send(requireToken(accessToken), input),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ternaryKeys.all }),
  });
};

export const useTernaryCreate = () =>
  useSettingsMutation<TernarySettingsInput>((accessToken, body) =>
    apiClient.post<TernaryMessageResponse>("/ternary/init", { accessToken, body }),
  );

export const useTernaryUpdate = () =>
  useSettingsMutation<TernarySettingsUpdate>((accessToken, body) =>
    apiClient.put<TernaryMessageResponse>("/ternary/settings", { accessToken, body }),
  );

export const useTernaryDelete = () =>
  useSettingsMutation<void>((accessToken) =>
    apiClient.delete<TernaryMessageResponse>("/ternary/delete", { accessToken }),
  );

export const useTernaryDryRun = () => {
  const { accessToken } = useAuthorized();
  return useMutation<TernaryDryRunResponse, Error, { limit: number }>({
    mutationFn: (body) =>
      apiClient.post<TernaryDryRunResponse>("/ternary/dry-run", { accessToken: requireToken(accessToken), body }),
  });
};

export const useTernaryExport = () => {
  const { accessToken } = useAuthorized();
  return useMutation<TernaryExportResponse, Error, TernaryExportRequest>({
    mutationFn: (body) =>
      apiClient.post<TernaryExportResponse>("/ternary/export", { accessToken: requireToken(accessToken), body }),
  });
};

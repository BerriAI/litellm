"use client";

import { useQuery } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { isProxyAdminTierRole } from "@/utils/roles";
import { createQueryKeys } from "../common/queryKeysFactory";

export type ModelDeprecationInfo = components["schemas"]["ModelDeprecationInfo"];
export type ModelDeprecationResponse = components["schemas"]["ModelDeprecationResponse"];

export const modelDeprecationKeys = createQueryKeys("modelDeprecations");

const fetchModelDeprecations = async (): Promise<ModelDeprecationResponse> => {
  const { data } = await backgroundFetchClient.GET("/model/deprecations");
  if (data === undefined) {
    throw new Error("The proxy returned no model deprecations payload");
  }
  return data;
};

export const useModelDeprecations = () => {
  const { accessToken, userRole } = useAuthorized();
  const queryOptions = {
    queryKey: modelDeprecationKeys.all,
    queryFn: fetchModelDeprecations,
    enabled: Boolean(accessToken) && isProxyAdminTierRole(userRole),
    staleTime: 5 * 60 * 1000,
    retry: false,
  };
  return useQuery(queryOptions);
};

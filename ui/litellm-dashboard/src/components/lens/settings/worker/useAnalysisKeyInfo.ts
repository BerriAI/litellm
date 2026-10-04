"use client";

import { useQuery } from "@tanstack/react-query";
import { analysisKeyInfoQuery } from "../../api/queries";
import { useLensApi } from "../../LensServices";

export function useAnalysisKeyInfo(accessToken: string, keyId?: string) {
  const api = useLensApi(accessToken);
  return useQuery(analysisKeyInfoQuery(api, keyId));
}

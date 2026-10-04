"use client";

import { useQuery } from "@tanstack/react-query";
import { analysisKeyInfoQuery } from "../../api/queries";
import { useLensApi } from "../../LensServices";

export function useAnalysisKeyInfo(keyId?: string) {
  const api = useLensApi();
  return useQuery(analysisKeyInfoQuery(api, keyId));
}

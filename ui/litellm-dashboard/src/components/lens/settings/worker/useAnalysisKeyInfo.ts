"use client";

import { useQuery } from "@tanstack/react-query";
import { analysisKeyInfoQuery } from "../../data/queries";
import { useLensApi } from "../../data/LensServices";

export function useAnalysisKeyInfo(keyId?: string) {
  const api = useLensApi();
  return useQuery(analysisKeyInfoQuery(api, keyId));
}

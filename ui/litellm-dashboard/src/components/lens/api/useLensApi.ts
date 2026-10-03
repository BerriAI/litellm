"use client";

import { apiClient } from "@/components/networking";
import { useLensDemo } from "../LensDemoContext";

export function useLensApi() {
  return useLensDemo()?.client ?? apiClient;
}

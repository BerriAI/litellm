import { fetchClient } from "@/lib/http/api";
import type { App } from "./appForms";

export const loadApps = async (offset = 0): Promise<App[]> => {
  const result = await fetchClient.GET("/management/v1/apps", { params: { query: { offset, limit: 100 } } });
  const page = result.data ?? [];
  return page.length === 100 ? [...page, ...(await loadApps(offset + 100))] : page;
};

export const appSpend = (spend: number) =>
  new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 8 }).format(spend);

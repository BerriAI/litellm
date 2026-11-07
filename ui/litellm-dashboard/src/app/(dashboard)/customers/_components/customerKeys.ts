import { useQuery } from "@tanstack/react-query";
import { keyListCall, keyUpdateCall, uiSpendLogsCall } from "@/components/networking";
import { monthRange } from "./customerPortalData";

export interface PortalKey {
  token: string;
  alias: string;
  models: string[];
  tpmLimit: number | null;
  rpmLimit: number | null;
  metadata: Record<string, unknown>;
}

const customerIds = (metadata: Record<string, unknown>): string[] => {
  const ids = metadata.customer_ids;
  return Array.isArray(ids) ? ids.filter((x): x is string => typeof x === "string") : [];
};

export const keyTaggedWith = (key: PortalKey, customerId: string): boolean => customerIds(key.metadata).includes(customerId);

export const useAllKeys = (accessToken: string | null) =>
  useQuery({
    queryKey: ["customer-portal", "keys"],
    enabled: Boolean(accessToken),
    queryFn: async (): Promise<PortalKey[]> => {
      const res = (await keyListCall(accessToken!, null, null, null, null, null, 1, 100)) as { keys?: unknown[] };
      return (res.keys ?? [])
        .filter((k): k is Record<string, unknown> => typeof k === "object" && k !== null)
        .map((k) => ({
          token: String(k.token ?? ""),
          alias: String(k.key_alias ?? k.token ?? ""),
          models: Array.isArray(k.models) ? (k.models as string[]) : [],
          tpmLimit: typeof k.tpm_limit === "number" ? k.tpm_limit : null,
          rpmLimit: typeof k.rpm_limit === "number" ? k.rpm_limit : null,
          metadata: (k.metadata && typeof k.metadata === "object" ? k.metadata : {}) as Record<string, unknown>,
        }));
    },
  });

export const useCustomerTrafficKeys = (accessToken: string | null, customerId: string) =>
  useQuery({
    queryKey: ["customer-portal", "traffic-keys", customerId],
    enabled: Boolean(accessToken) && Boolean(customerId),
    queryFn: async (): Promise<Set<string>> => {
      const { start, end } = monthRange(0);
      const fmt = (d: Date) => d.toISOString().slice(0, 19).replace("T", " ");
      const res = (await uiSpendLogsCall({ accessToken: accessToken!, start_date: fmt(start), end_date: fmt(end), page: 1, page_size: 500, params: { end_user: customerId } })) as { data?: { api_key?: string }[] };
      return new Set((res.data ?? []).map((r) => r.api_key ?? "").filter(Boolean));
    },
  });

export const setCustomerTag = async (accessToken: string, key: PortalKey, customerId: string, tagged: boolean) => {
  const others = customerIds(key.metadata).filter((id) => id !== customerId);
  const ids = tagged ? [...others, customerId] : others;
  await keyUpdateCall(accessToken, { key: key.token, metadata: { ...key.metadata, customer_ids: ids } });
};

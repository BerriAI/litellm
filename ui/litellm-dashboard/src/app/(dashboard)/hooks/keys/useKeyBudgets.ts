import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type KeyBudgetsResponse = components["schemas"]["KeyBudgetsResponse"];
export type KeyBudgetEntry = KeyBudgetsResponse["budgets"][number];

export const useKeyBudgets = (keyId: string | undefined) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    "/key/{key_id}/budgets",
    { params: { path: { key_id: keyId ?? "" } } },
    { enabled: Boolean(accessToken) && Boolean(keyId) },
  );
};

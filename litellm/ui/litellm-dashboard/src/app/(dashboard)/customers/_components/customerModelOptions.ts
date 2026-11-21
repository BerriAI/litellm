import { fetchClient } from "@/lib/http/api";
import type { AppResource } from "../../apps/_components/appForms";

export type CustomerModelScope = Pick<AppResource, "resource_type" | "resource_id">;

export async function customerModelOptions(scope: CustomerModelScope) {
  const { data, error } = await fetchClient.GET("/management/v1/apps/resources/{resource_type}/{resource_id}/models", {
    params: { path: scope },
  });
  if (error || !data) throw new Error("Could not load app models");
  return { data: data.map((id) => ({ id, owned_by: "" })) };
}

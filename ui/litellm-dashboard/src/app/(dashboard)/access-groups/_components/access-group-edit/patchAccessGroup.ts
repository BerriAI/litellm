import type { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { fetchClient } from "@/lib/http/api";

import type { AccessGroupPatchBody } from "./mapper";

export const patchAccessGroup = async (
  accessGroupId: string,
  body: AccessGroupPatchBody,
): Promise<AccessGroupResponse | undefined> => {
  const { data } = await fetchClient.PATCH("/management/v1/access-groups/{access_group_id}", {
    params: { path: { access_group_id: accessGroupId } },
    body,
  });
  return data?.data;
};

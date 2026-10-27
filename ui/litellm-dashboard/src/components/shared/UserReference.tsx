"use client";

import { useUserLookup } from "@/app/(dashboard)/hooks/users/useUsers";
import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";
import { EntityLink } from "@/components/shared/EntityLink";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";
import { userDetailHref } from "@/utils/entityLinks";

export function UserReference({ userId }: { userId: string }) {
  const isDefaultAdmin = userId === DEFAULT_PROXY_ADMIN_USER_ID;
  const { data: user } = useUserLookup(isDefaultAdmin ? null : userId);

  if (isDefaultAdmin) {
    return <DefaultProxyAdminTag userId={userId} />;
  }

  const name = user?.user_alias?.trim() || user?.user_email?.trim() || null;
  const href = userDetailHref(userId);

  if (!name) {
    return <EntityLink href={href}>{userId}</EntityLink>;
  }

  return (
    <SimpleTooltip content={userId}>
      <EntityLink href={href}>{name}</EntityLink>
    </SimpleTooltip>
  );
}

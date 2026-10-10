"use client";

import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";
import { EntityLink } from "@/components/shared/EntityLink";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { userDetailHref } from "@/utils/entityLinks";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";

interface EntityReferenceProps {
  id: string;
  name?: string | null;
  href?: string;
}

export function EntityReference({ id, name, href }: EntityReferenceProps) {
  if (!name) {
    return (
      <EntityLink href={href} className="font-mono text-xs font-normal">
        {id}
      </EntityLink>
    );
  }
  return (
    <SimpleTooltip content={id} className="min-w-0 max-w-full">
      <EntityLink href={href}>{name}</EntityLink>
    </SimpleTooltip>
  );
}

interface UserReferenceProps {
  userId: string | null | undefined;
  displayName?: string | null;
}

export function UserReference({ userId, displayName }: UserReferenceProps) {
  if (userId == null || userId === "") return null;
  if (userId === DEFAULT_PROXY_ADMIN_USER_ID && !displayName) {
    return <DefaultProxyAdminTag userId={userId} />;
  }
  return <EntityReference id={userId} name={displayName} href={userDetailHref(userId)} />;
}

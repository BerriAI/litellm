"use client";

import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { userDetailHref } from "@/utils/entityLinks";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";

import { IdCell } from "./id_cell";
import { IdentityCell } from "./identity_cell";

export const ENTITY_CELL_TITLE_CLASSES = "font-mono text-xs font-normal";

interface UserPopoverCellProps {
  userAlias: string | null;
  userEmail: string | null;
  userId: string | null;
  width: number;
}

export function UserPopoverCell({ userAlias, userEmail, userId, width }: UserPopoverCellProps) {
  const displayValue = userAlias || userEmail || userId;
  const isDefaultAdmin = userId === DEFAULT_PROXY_ADMIN_USER_ID;

  const popoverContent = (
    <div className="flex flex-col gap-2 text-xs min-w-[200px] max-w-[300px]">
      {[
        { label: "User Alias", value: userAlias },
        { label: "User Email", value: userEmail },
        { label: "User ID", value: userId },
      ].map(({ label, value }) => (
        <div key={label} className="flex flex-col min-w-0">
          <span className="text-muted-foreground">{label}</span>
          {value ? (
            <IdCell value={value} variant="plain" copyable copyLabel={`Copy ${label}`} className="max-w-full" />
          ) : (
            <span className="font-mono">-</span>
          )}
        </div>
      ))}
    </div>
  );

  const trigger =
    isDefaultAdmin && !userAlias && !userEmail ? (
      <DefaultProxyAdminTag userId={userId} />
    ) : (
      <IdentityCell
        title={displayValue || "-"}
        titleClassName={ENTITY_CELL_TITLE_CLASSES}
        href={userId ? userDetailHref(userId) : undefined}
      />
    );

  return (
    <HoverCard>
      <HoverCardTrigger render={<span className="block" style={{ maxWidth: width, overflow: "hidden" }} />}>
        {trigger}
      </HoverCardTrigger>
      <HoverCardContent align="start">{popoverContent}</HoverCardContent>
    </HoverCard>
  );
}

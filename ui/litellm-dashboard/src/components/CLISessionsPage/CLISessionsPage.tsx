"use client";

import { PaginationState } from "@tanstack/react-table";
import { Terminal } from "lucide-react";
import { useCallback, useState } from "react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCLISessions, useRevokeCLISession } from "@/app/(dashboard)/hooks/cliSessions/useCLISessions";
import { PageHeader } from "@/components/shared/PageHeader";
import { isProxyAdminRole } from "@/utils/roles";

import { CLISessionsTable } from "./CLISessionsTable/CLISessionsTable";

export default function CLISessionsPage() {
  // `effectiveSessionRole` normalizes proxy_admin_viewer to "Admin" for read parity,
  // so the role alone cannot tell a revoker from a reader; isViewOnly can.
  const { userRole, isViewOnly } = useAuthorized();
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });

  const { data, isLoading } = useCLISessions(pagination.pageIndex + 1, pagination.pageSize);
  const revoke = useRevokeCLISession();

  const onRevoke = useCallback(
    (sessionId: string) => revoke.mutate({ params: { path: { session_id: sessionId } } }),
    [revoke],
  );

  return (
    <div className="flex flex-col gap-4">
      <PageHeader
        title="CLI Sessions"
        subtitle="Credentials handed out by `lite login`. Revoking one cuts it off before it expires."
        icon={<Terminal className="size-5" />}
      />
      <CLISessionsTable
        sessions={data?.sessions || []}
        totalCount={data?.total_count || 0}
        isLoading={isLoading}
        isRevoking={revoke.isPending}
        canRevoke={isProxyAdminRole(userRole || "") && !isViewOnly}
        onRevoke={onRevoke}
        pagination={pagination}
        onPaginationChange={setPagination}
      />
    </div>
  );
}

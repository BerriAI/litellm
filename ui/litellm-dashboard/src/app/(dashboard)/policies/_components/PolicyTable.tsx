"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { Policy } from "@/components/policies/types";

import { getPolicyTableColumns, PolicyRow } from "./PolicyTableColumns";

/** One row per policy name; primaryPolicy is used for display and for Edit (FlowBuilder loads all versions) */
function groupPoliciesByName(policies: Policy[]): PolicyRow[] {
  const names = Array.from(new Set(policies.map((policy) => policy.policy_name || "(unnamed)")));
  return names.map((policyName) => {
    const versions = policies.filter((policy) => (policy.policy_name || "(unnamed)") === policyName);
    const primary =
      versions.find((version) => version.version_status === "production") ??
      [...versions].sort((a, b) => (b.version_number ?? 0) - (a.version_number ?? 0))[0];
    return { policy_name: policyName, primaryPolicy: primary, versionCount: versions.length };
  });
}

interface PolicyTableProps {
  policies: Policy[];
  isLoading: boolean;
  onDeleteClick: (policyId: string, policyName: string) => void;
  onEditClick: (policy: Policy) => void;
  onViewClick: (policyId: string) => void;
  isAdmin?: boolean;
}

const DEFAULT_SORTING: SortingState = [{ id: "policy_name", desc: false }];

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No policies found</div>
      <div className="text-sm text-muted-foreground">
        Create a policy to bundle guardrails and apply them across teams.
      </div>
    </div>
  );
}

const PolicyTable: React.FC<PolicyTableProps> = ({
  policies,
  isLoading,
  onDeleteClick,
  onEditClick,
  onViewClick,
  isAdmin = false,
}) => {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const rows = useMemo(() => groupPoliciesByName(policies), [policies]);

  const columns = useMemo(() => {
    const deps = { isAdmin, onViewClick, onEditClick, onDeleteClick };
    return getPolicyTableColumns(deps);
  }, [isAdmin, onViewClick, onEditClick, onDeleteClick]);

  return (
    <DataTable
      data={rows}
      columns={columns}
      getRowId={(row) => row.policy_name}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage="Loading policies…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default PolicyTable;

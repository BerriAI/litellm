"use client";

import { SortingState } from "@tanstack/react-table";
import { Users } from "lucide-react";
import React, { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";

import { AvailableTeam, getAvailableTeamsTableColumns } from "./AvailableTeamsTableColumns";

interface AvailableTeamsTableProps {
  teams: AvailableTeam[];
  isLoading: boolean;
  onJoinTeam: (teamId: string) => void;
}

const DEFAULT_SORTING: SortingState = [{ id: "team_alias", desc: false }];

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Users className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No available teams to join</div>
      <div className="text-sm text-muted-foreground">
        See how to set available teams{" "}
        <a
          href="https://docs.litellm.ai/docs/proxy/self_serve#all-settings-for-self-serve--sso-flow"
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline-offset-4 hover:underline"
        >
          here
        </a>
      </div>
    </div>
  );
}

const AvailableTeamsTable: React.FC<AvailableTeamsTableProps> = ({ teams, isLoading, onJoinTeam }) => {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => getAvailableTeamsTableColumns({ onJoinTeam }), [onJoinTeam]);

  return (
    <DataTable
      data={teams}
      columns={columns}
      getRowId={(team, index) => team.team_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage="Loading available teams…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default AvailableTeamsTable;

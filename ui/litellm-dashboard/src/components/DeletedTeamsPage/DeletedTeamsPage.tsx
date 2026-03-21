"use client";
import { useDeletedTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { DeletedTeamsTable } from "./DeletedTeamsTable/DeletedTeamsTable";

export default function DeletedTeamsPage() {
  const {
    data: teamsData,
    isPending: isLoading,
    isFetching,
  } = useDeletedTeams(1, 100);

  return (
    <DeletedTeamsTable
      teams={teamsData || []}
      isLoading={isLoading}
      isFetching={isFetching}
    />
  );
}

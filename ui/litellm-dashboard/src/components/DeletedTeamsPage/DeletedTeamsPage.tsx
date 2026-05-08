"use client";
import { Alert } from "antd";
import { useDeletedTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedTeamsTable } from "./DeletedTeamsTable/DeletedTeamsTable";

export default function DeletedTeamsPage() {
  const { premiumUser } = useAuthorized();
  const {
    data: teamsData,
    isPending: isLoading,
    isFetching,
  } = useDeletedTeams(1, 100);

  return (
    <div className="flex flex-col gap-4">
      {!premiumUser && (
        <Alert
          type="info"
          banner
          showIcon
          message="Coming soon to Enterprise"
          description="Deleted team auditing is graduating from beta into our Enterprise audit & compliance suite."
        />
      )}
      <DeletedTeamsTable
        teams={teamsData || []}
        isLoading={isLoading}
        isFetching={isFetching}
      />
    </div>
  );
}

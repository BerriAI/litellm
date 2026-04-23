"use client";
import { Info } from "lucide-react";
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
        <div className="flex gap-2 items-start p-3 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 text-blue-800 dark:text-blue-200">
          <Info className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <div className="font-semibold">Coming soon to Enterprise</div>
            <div className="text-sm">
              Deleted team auditing is graduating from beta into our Enterprise
              audit &amp; compliance suite.
            </div>
          </div>
        </div>
      )}
      <DeletedTeamsTable
        teams={teamsData || []}
        isLoading={isLoading}
        isFetching={isFetching}
      />
    </div>
  );
}

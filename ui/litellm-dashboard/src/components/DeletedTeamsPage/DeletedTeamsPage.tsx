"use client";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useDeletedTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DeletedTeamsTable } from "./DeletedTeamsTable/DeletedTeamsTable";

export default function DeletedTeamsPage() {
  const { premiumUser } = useAuthorized();
  const { data: teamsData, isLoading } = useDeletedTeams(1, 100);

  return (
    <div className="flex flex-col gap-4">
      {!premiumUser && (
        <Alert>
          <Info />
          <AlertTitle>Coming soon to Enterprise</AlertTitle>
          <AlertDescription>
            Deleted team auditing is graduating from beta into our Enterprise audit &amp; compliance suite.
          </AlertDescription>
        </Alert>
      )}
      <DeletedTeamsTable teams={teamsData || []} isLoading={isLoading} />
    </div>
  );
}

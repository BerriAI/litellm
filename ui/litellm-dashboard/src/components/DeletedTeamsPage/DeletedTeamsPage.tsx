"use client";
import { Alert } from "antd";
import { useDeletedTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTranslation } from "react-i18next";
import { DeletedTeamsTable } from "./DeletedTeamsTable/DeletedTeamsTable";

export default function DeletedTeamsPage() {
  const { t } = useTranslation();
  const { premiumUser } = useAuthorized();
  const { data: teamsData, isPending: isLoading, isFetching } = useDeletedTeams(1, 100);

  return (
    <div className="flex flex-col gap-4">
      {!premiumUser && (
        <Alert
          type="info"
          banner
          showIcon
          message={t("deletedTeams.deletedTeamsPage.comingSoonToEnterprise")}
          description={t("deletedTeams.deletedTeamsPage.enterpriseAuditDescription")}
        />
      )}
      <DeletedTeamsTable teams={teamsData || []} isLoading={isLoading} isFetching={isFetching} />
    </div>
  );
}

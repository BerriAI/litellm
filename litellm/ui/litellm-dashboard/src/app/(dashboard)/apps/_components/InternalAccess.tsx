"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "@/lib/toast";
import useAuthorized from "../../hooks/useAuthorized";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import TeamDropdown from "@/components/common_components/team_dropdown";
import { teamInfoCall, teamUpdateCall } from "@/components/networking";
import AppAccessSelector from "./AppAccessSelector";
import { appAccessChanged, teamAppIds, type TeamAppAccess } from "./internalAccess";
import { grantedAppsKey, readsEveryApp } from "./useGrantedApps";

function TeamAppAccessEditor({ teamId, accessToken }: { teamId: string; accessToken: string }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const queryKey = ["apps", "internal-access", teamId] as const;
  const team = useQuery({
    queryKey,
    queryFn: async (): Promise<TeamAppAccess> => await teamInfoCall(accessToken, teamId, { includeMembers: false }),
  });
  const [selected, setSelected] = useState<string[] | null>(null);
  const stored = teamAppIds(team.data);
  const value = selected ?? stored;
  const save = useMutation({
    mutationFn: () => teamUpdateCall(accessToken, { team_id: teamId, object_permission: { apps: value } }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey });
      await queryClient.invalidateQueries({ queryKey: grantedAppsKey });
      setSelected(null);
      toast.success(t("apps.internalAccess.saved"));
    },
  });
  if (team.isError) return <ErrorState onRetry={() => void team.refetch()} />;
  if (team.isPending) return <p>{t("customerPortal.common.loading")}</p>;
  return (
    <div className="space-y-3">
      <label htmlFor="internal-access-apps" className="text-sm font-medium">
        {t("apps.internalAccess.apps")}
      </label>
      <AppAccessSelector
        id="internal-access-apps"
        value={value}
        onChange={setSelected}
        placeholder={t("apps.internalAccess.appsPlaceholder")}
      />
      <Button disabled={save.isPending || !appAccessChanged(stored, value)} onClick={() => save.mutate()}>
        {t("apps.internalAccess.save")}
      </Button>
    </div>
  );
}

export default function InternalAccess() {
  const { t } = useTranslation();
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const [teamId, setTeamId] = useState<string | null>(null);
  if (!accessToken || !readsEveryApp(userRole, isViewOnly)) return <AdminOnlyNotice pageTitle={t("apps.internalAccess.title")} />;
  return (
    <main className="max-w-2xl space-y-5 px-6 py-5">
      <h1 className="text-lg font-semibold">{t("apps.internalAccess.title")}</h1>
      <p className="text-sm text-muted-foreground">{t("apps.internalAccess.hint")}</p>
      <div className="space-y-3">
        <label htmlFor="internal-access-team" className="text-sm font-medium">
          {t("apps.internalAccess.team")}
        </label>
        <TeamDropdown id="internal-access-team" value={teamId} onChange={setTeamId} />
      </div>
      {teamId && <TeamAppAccessEditor key={teamId} teamId={teamId} accessToken={accessToken} />}
    </main>
  );
}

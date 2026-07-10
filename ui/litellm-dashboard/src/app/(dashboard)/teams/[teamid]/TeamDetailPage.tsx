"use client";

import { useAllTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { fetchAvailableModelsForTeamOrKey } from "@/components/key_team_helpers/fetch_available_models_team_key";
import TeamInfoView from "@/components/team/TeamInfo";
import { migratedHref } from "@/utils/migratedPages";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "antd";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

function teamIdFromPathname(pathname: string): string {
  const segments = pathname.replace(/\/+$/, "").split("/");
  return decodeURIComponent(segments[segments.length - 1] ?? "");
}

export default function TeamDetailPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { isLoading: authLoading, isAuthorized, accessToken, userId, userRole, premiumUser } = useAuthorized();
  const [teamId] = useState(() => (typeof window === "undefined" ? "" : teamIdFromPathname(window.location.pathname)));
  const [userModels, setUserModels] = useState<string[]>([]);

  const { data: teams, isPending } = useAllTeams();

  useEffect(() => {
    if (!accessToken || !userId || !userRole) {
      return;
    }
    fetchAvailableModelsForTeamOrKey(userId, userRole, accessToken)
      .then((models) => {
        if (models) {
          setUserModels(models);
        }
      })
      .catch((error) => console.error("Error fetching user models:", error));
  }, [accessToken, userId, userRole]);

  const backToTeams = () => router.push(migratedHref("teams"));

  if (authLoading || !isAuthorized) {
    return <LoadingScreen />;
  }
  if (!teamId || isPending) {
    return <LoadingScreen />;
  }

  const team = teams?.find((t) => t.team_id === teamId);
  if (!team) {
    return (
      <div className="p-4">
        <Button type="text" icon={<ArrowLeftOutlined />} onClick={backToTeams} className="mb-4">
          Back to Teams
        </Button>
        <p className="text-sm text-gray-700">Team not found</p>
      </div>
    );
  }

  const isTeamAdmin = (team.members_with_roles ?? []).some((m) => m.user_id === userId && m.role === "admin");

  return (
    <TeamInfoView
      teamId={teamId}
      accessToken={accessToken}
      is_team_admin={isTeamAdmin}
      is_proxy_admin={userRole === "Admin"}
      userModels={userModels}
      editTeam={false}
      premiumUser={premiumUser ?? undefined}
      onClose={backToTeams}
      onUpdate={() => queryClient.invalidateQueries({ queryKey: ["teams"] })}
    />
  );
}

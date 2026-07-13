import { Select, Typography } from "antd";
import React, { useEffect } from "react";

import { Team } from "../../key_team_helpers/key_list";
import MyUserTab from "../../team/MyUserTab";

interface MyBudgetsUsageViewProps {
  teams: Team[];
  selectedTeamId: string | null;
  onTeamChange: (teamId: string) => void;
}

export default function MyBudgetsUsageView({ teams, selectedTeamId, onTeamChange }: MyBudgetsUsageViewProps) {
  const didInitTeamRef = React.useRef(false);

  useEffect(() => {
    if (teams.length === 0) {
      return;
    }
    if (selectedTeamId && teams.some((team) => team.team_id === selectedTeamId)) {
      didInitTeamRef.current = true;
      return;
    }
    // Only auto-pick once per mount when URL has no valid team. Avoid re-firing
    // when the parent clears team while navigating away from this view.
    if (didInitTeamRef.current) {
      return;
    }
    didInitTeamRef.current = true;
    onTeamChange(teams[0].team_id);
  }, [teams, selectedTeamId, onTeamChange]);

  if (teams.length === 0) {
    return (
      <Typography.Text type="secondary">
        You are not on any teams yet. Once you join a team with a per-user budget, your usage against that budget will
        show here.
      </Typography.Text>
    );
  }

  const activeTeamId =
    selectedTeamId && teams.some((team) => team.team_id === selectedTeamId) ? selectedTeamId : teams[0].team_id;

  return (
    <div className="space-y-4">
      <div>
        <Typography.Text type="secondary" className="block mb-2">
          Team
        </Typography.Text>
        <Select
          className="w-full max-w-md"
          value={activeTeamId}
          onChange={onTeamChange}
          options={teams.map((team) => ({
            value: team.team_id,
            label: team.team_alias || team.team_id,
          }))}
        />
      </div>
      <MyUserTab teamId={activeTeamId} />
    </div>
  );
}

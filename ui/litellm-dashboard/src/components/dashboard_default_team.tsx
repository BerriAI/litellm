import React, { useState, useEffect } from "react";
import { Select, SelectItem, Text, Title } from "@tremor/react";

interface DashboardTeamProps {
  teams: Object[] | null;
  setSelectedTeam: React.Dispatch<React.SetStateAction<any | null>>;
  userRole: string | null;
}

type TeamInterface = {
  models: any[];
  team_id: null;
  team_alias: String
}

const DashboardTeam: React.FC<DashboardTeamProps> = ({
  teams,
  setSelectedTeam,
  userRole,
}) => {
  const defaultTeam: TeamInterface = {
    models: [],
    team_id: null,
    team_alias: "Default Team"
  }


  const [value, setValue] = useState(defaultTeam);

  let updatedTeams;
  if (userRole === "App User") {
    // Non-Admin SSO users should only see their own team - they should not see "Default Team"
    updatedTeams = teams;
  } else {
    updatedTeams = teams ? [...teams, defaultTeam] : [defaultTeam];
  }
  if (userRole === 'App User') return null;

  return (
    <div className="mt-5 mb-5">
      <Title>Select Team</Title>
      
      <Text>
        If you belong to multiple teams, this setting controls which team is used by default when creating new API Keys.
      </Text>
      <Text className="mt-3 mb-3">
        <b>Default Team:</b> If no team_id is set for a key, it will be grouped under here.
      </Text>

      {updatedTeams && updatedTeams.length > 0 ? (
        <Select defaultValue="0">
          {updatedTeams.map((team: any, index) => (
            <SelectItem
              key={index}
              value={String(index)}
              onClick={() => setSelectedTeam(team)}
            >
              {team["team_alias"]}
            </SelectItem>
          ))}
        </Select>
      ) : (
        <Text>
          No team created. <b>Defaulting to personal account.</b>
        </Text>
      )}
    </div>
  );
};

export default DashboardTeam;

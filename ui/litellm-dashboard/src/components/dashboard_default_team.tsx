import React, { useState, useEffect } from "react";
import { Select, SelectItem, Text, Title } from "@tremor/react";

interface DashboardTeamProps {
  teams: Object[] | null;
  setSelectedTeam: React.Dispatch<React.SetStateAction<any | null>>;
}

const DashboardTeam: React.FC<DashboardTeamProps> = ({
  teams,
  setSelectedTeam,
}) => {
  const [value, setValue] = useState("");

  return (
    <div className="mt-5 mb-5">
      <Title>Select Team</Title>
      <Text>
        If you belong to multiple teams, this setting controls which team is
        used by default when creating new API Keys.
      </Text>
      {teams && teams.length > 0 ? (
        <Select defaultValue="0">
          {teams.map((team: any, index) => (
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

import React, { useState, useEffect } from "react";
import { Typography } from "antd";
import { Select, SelectItem } from "@tremor/react";

interface DashboardTeamProps {
  teams: Object[] | null;
  setSelectedTeam: React.Dispatch<React.SetStateAction<any | null>>;
}

const DashboardTeam: React.FC<DashboardTeamProps> = ({
  teams,
  setSelectedTeam,
}) => {
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");

  return (
    <div className="mt-10">
      <Title level={4}>Default Team</Title>
      <Paragraph>
        If you belong to multiple teams, this setting controls which team is
        used by default when creating new API Keys.
      </Paragraph>
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
        <Paragraph>
          No team created. <b>Defaulting to personal account.</b>
        </Paragraph>
      )}
    </div>
  );
};

export default DashboardTeam;

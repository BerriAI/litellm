import React, { useState, useEffect } from "react";
import { Typography } from "antd";
import { Select, SelectItem } from "@tremor/react";

interface DashboardTeamProps {
  teams: string[] | null;
}

const DashboardTeam: React.FC<DashboardTeamProps> = ({ teams }) => {
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");
  console.log(`received teams ${teams}`);
  return (
    <div className="mt-10">
      <Title level={4}>Default Team</Title>
      <Paragraph>
        If you belong to multiple teams, this setting controls which
        organization is used by default when creating new API Keys.
      </Paragraph>
      {teams && teams.length > 0 ? (
        <Select
          id="distance"
          value={value}
          onValueChange={setValue}
          className="mt-2"
        >
          {teams.map((model) => (
            <SelectItem value="model">{model}</SelectItem>
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

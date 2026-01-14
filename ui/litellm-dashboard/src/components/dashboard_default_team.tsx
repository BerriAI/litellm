import React, { useState, useEffect } from "react";
import { Select, SelectItem, Text, Title } from "@tremor/react";
import { ProxySettings, UserInfo } from "./user_dashboard";
import { getProxyUISettings } from "./networking";

interface DashboardTeamProps {
  teams: object[] | null;
  setSelectedTeam: React.Dispatch<React.SetStateAction<any | null>>;
  userRole: string | null;
  proxySettings: ProxySettings | null;
  setProxySettings: React.Dispatch<React.SetStateAction<ProxySettings | null>>;
  userInfo: UserInfo | null;
  accessToken: string | null;
  setKeys: React.Dispatch<React.SetStateAction<any | null>>;
}

type TeamInterface = {
  models: any[];
  team_id: null;
  team_alias: string;
  max_budget: number | null;
};

const DashboardTeam: React.FC<DashboardTeamProps> = ({
  teams,
  setSelectedTeam,
  userRole,
  proxySettings,
  setProxySettings,
  userInfo,
  accessToken,
  setKeys,
}) => {
  console.log(`userInfo: ${JSON.stringify(userInfo)}`);
  const defaultTeam: TeamInterface = {
    models: userInfo?.models || [],
    team_id: null,
    team_alias: "Default Team",
    max_budget: userInfo?.max_budget || null,
  };

  const getProxySettings = async () => {
    if (proxySettings === null && accessToken) {
      const proxy_settings: ProxySettings = await getProxyUISettings(accessToken);
      setProxySettings(proxy_settings);
    }
  };

  useEffect(() => {
    getProxySettings();
  }, [proxySettings]);

  const [value, setValue] = useState(defaultTeam);

  let updatedTeams;
  console.log(`userRole: ${userRole}`);
  console.log(`proxySettings: ${JSON.stringify(proxySettings)}`);
  if (userRole === "App User") {
    // Non-Admin SSO users should only see their own team - they should not see "Default Team"
    updatedTeams = teams;
  } else if (proxySettings && proxySettings.DEFAULT_TEAM_DISABLED === true) {
    updatedTeams = teams ? [...teams] : [defaultTeam];
  } else {
    updatedTeams = teams ? [...teams, defaultTeam] : [defaultTeam];
  }

  return (
    <div className="mt-5 mb-5">
      <Title>Select Team</Title>

      <Text>
        If you belong to multiple teams, this setting controls which team is used by default when creating new Virtual
        Keys.
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
              onClick={() => {
                setSelectedTeam(team);
                // setKeys(team["keys"]);
              }}
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

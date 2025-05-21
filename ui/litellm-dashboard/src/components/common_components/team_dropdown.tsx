import React from "react";
import { Select } from "antd";
import { Team } from "../key_team_helpers/key_list";

interface TeamDropdownProps {
  teams?: Team[] | null;
  value?: string;
  onChange?: (value: string) => void;
}

const TeamDropdown: React.FC<TeamDropdownProps> = ({ teams, value, onChange }) => {
  return (
    <Select
      showSearch
      placeholder="Search or select a team"
      value={value}
      onChange={onChange}
      filterOption={(input, option) => {
        if (!option) return false;
        const teamAlias = option.children?.[0]?.props?.children || '';
        return teamAlias.toLowerCase().includes(input.toLowerCase());
      }}
      optionFilterProp="children"
    >
      {teams?.map((team) => (
        <Select.Option key={team.team_id} value={team.team_id}>
          <span className="font-medium">{team.team_alias}</span>{" "}
          <span className="text-gray-500">({team.team_id})</span>
        </Select.Option>
      ))}
    </Select>
  );
};

export default TeamDropdown; 
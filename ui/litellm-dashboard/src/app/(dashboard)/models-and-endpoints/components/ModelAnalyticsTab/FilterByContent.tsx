import { Select, SelectItem, Text } from "@tremor/react";
import React, { useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Team } from "@/components/key_team_helpers/key_list";

interface FilterByContentProps {
  setSelectedAPIKey: (key: any) => void;
  keys: any[] | null;
  teams: Team[] | null;
  setSelectedCustomer: (customer: string | null) => void;
  allEndUsers: any[];
}

const FilterByContent = ({
  setSelectedAPIKey,
  keys,
  teams,
  setSelectedCustomer,
  allEndUsers,
}: FilterByContentProps) => {
  const { premiumUser } = useAuthorized();

  const [selectedTeamFilter, setSelectedTeamFilter] = useState<string | null>(null);

  return (
    <div>
      <Text className="mb-1">Select API Key Name</Text>

      {premiumUser ? (
        <div>
          <Select defaultValue="all-keys">
            <SelectItem
              key="all-keys"
              value="all-keys"
              onClick={() => {
                setSelectedAPIKey(null);
              }}
            >
              All Keys
            </SelectItem>
            {keys?.map((key: any, index: number) => {
              if (key && key["key_alias"] !== null && key["key_alias"].length > 0) {
                return (
                  <SelectItem
                    key={index}
                    value={String(index)}
                    onClick={() => {
                      setSelectedAPIKey(key);
                    }}
                  >
                    {key["key_alias"]}
                  </SelectItem>
                );
              }
              return null;
            })}
          </Select>

          <Text className="mt-1">Select Customer Name</Text>

          <Select defaultValue="all-customers">
            <SelectItem
              key="all-customers"
              value="all-customers"
              onClick={() => {
                setSelectedCustomer(null);
              }}
            >
              All Customers
            </SelectItem>
            {allEndUsers?.map((user: any, index: number) => {
              return (
                <SelectItem
                  key={index}
                  value={user}
                  onClick={() => {
                    setSelectedCustomer(user);
                  }}
                >
                  {user}
                </SelectItem>
              );
            })}
          </Select>

          <Text className="mt-1">Select Team</Text>

          <Select
            className="w-64 relative z-50"
            defaultValue="all"
            value={selectedTeamFilter ?? "all"}
            onValueChange={(value) => setSelectedTeamFilter(value === "all" ? null : value)}
          >
            <SelectItem value="all">All Teams</SelectItem>
            {teams
              ?.filter((team) => team.team_id)
              .map((team) => (
                <SelectItem key={team.team_id} value={team.team_id}>
                  {team.team_alias
                    ? `${team.team_alias} (${team.team_id.slice(0, 8)}...)`
                    : `Team ${team.team_id.slice(0, 8)}...`}
                </SelectItem>
              ))}
          </Select>
        </div>
      ) : (
        <div>
          {/* ... existing non-premium user content ... */}
          <Text className="mt-1">Select Team</Text>

          <Select
            className="w-64 relative z-50"
            defaultValue="all"
            value={selectedTeamFilter ?? "all"}
            onValueChange={(value) => setSelectedTeamFilter(value === "all" ? null : value)}
          >
            <SelectItem value="all">All Teams</SelectItem>
            {teams
              ?.filter((team) => team.team_id)
              .map((team) => (
                <SelectItem key={team.team_id} value={team.team_id}>
                  {team.team_alias
                    ? `${team.team_alias} (${team.team_id.slice(0, 8)}...)`
                    : `Team ${team.team_id.slice(0, 8)}...`}
                </SelectItem>
              ))}
          </Select>
        </div>
      )}
    </div>
  );
};

export default FilterByContent;

import { useState } from 'react';
import { Select, SelectItem, Text } from "@tremor/react";

interface ModelData {
  team_id: string;
  team_name: string;
  // Add other properties as needed
}

interface ModelDashboardProps {
  modelData: ModelData[];
}

export default function ModelDashboard({ modelData }: ModelDashboardProps) {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);

  const getTeamName = (teamId: string): string => {
    const team = modelData.find(item => item.team_id === teamId);
    return team?.team_name || 'Unknown Team';
  };

  return (
    <div className="flex flex-col space-y-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Text className="text-lg font-medium">Model Management</Text>
          <Text className="text-gray-500">Add and manage models for the proxy</Text>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Text>Filter by Public Model Name:</Text>
            <Select
              className="w-64"
              defaultValue="all"
            >
              <SelectItem value="all">All Models</SelectItem>
              {/* Add model options here */}
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <Text>Filter by Team:</Text>
            <Select
              className="w-64"
              value={selectedTeam ?? "all"}
              onValueChange={(value) => setSelectedTeam(value === "all" ? null : value)}
            >
              <SelectItem value="all">All Teams</SelectItem>
              {Array.from(new Set(modelData.map(model => model.team_id)))
                .filter(teamId => teamId !== null)
                .map(teamId => (
                  <SelectItem key={teamId} value={teamId}>
                    {getTeamName(teamId)}
                  </SelectItem>
                ))}
            </Select>
          </div>
        </div>
      </div>
    </div>
  );
} 
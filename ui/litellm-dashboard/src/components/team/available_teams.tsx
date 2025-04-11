import React, { useState, useEffect } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Button,
  Text,
  Badge,
  Select,
  SelectItem,
} from "@tremor/react";
import { message } from 'antd';
import { availableTeamListCall, teamMemberAddCall, addPublicTeamCall, deletePublicTeamCall } from "../networking";
import { fetchTeams } from "../common_components/fetch_teams";

interface AvailableTeam {
  team_id: string;
  team_alias: string;
  description?: string;
  models: string[];
  members_with_roles: {user_id?: string, user_email?: string, role: string}[];
}

interface AvailableTeamsProps {
  accessToken: string | null;
  userID: string | null;
}

const AvailableTeamsPanel: React.FC<AvailableTeamsProps> = ({
  accessToken,
  userID,
}) => {
  const [availableTeams, setAvailableTeams] = useState<AvailableTeam[]>([]);
  const [allTeams, setAllTeams] = useState<any[]>([]);
  const [selectedTeamId, setSelectedTeamId] = useState<string>("");

  useEffect(() => {
    const fetchData = async () => {
      if (!accessToken || !userID) return;
      
      try {
        // Fetch available (public) teams
        const availableResponse = await availableTeamListCall(accessToken);
        setAvailableTeams(availableResponse);

        // Fetch all teams for the dropdown
        await fetchTeams(accessToken, userID, "Admin", null, setAllTeams);
      } catch (error) {
        console.error('Error fetching teams:', error);
      }
    };

    fetchData();
  }, [accessToken, userID]);

  const handleJoinTeam = async (teamId: string) => {
    if (!accessToken || !userID) return;

    try {
      const response = await teamMemberAddCall(accessToken, teamId, {
        "user_id": userID,
        "role": "user"
      });
      
      message.success('Successfully joined team');
      // Update available teams list
      setAvailableTeams(teams => teams.filter(team => team.team_id !== teamId));
    } catch (error) {
      console.error('Error joining team:', error);
      message.error('Failed to join team');
    }
  };

  const handleAddPublicTeam = async () => {
    if (!accessToken || !selectedTeamId) {
      message.error('Please select a team to make public');
      return;
    }

    try {
      await addPublicTeamCall(accessToken, selectedTeamId);
      message.success('Successfully made team public');
      
      // Refresh both lists
      const availableResponse = await availableTeamListCall(accessToken);
      setAvailableTeams(availableResponse);
      await fetchTeams(accessToken, userID, "Admin", null, setAllTeams);
      
      setSelectedTeamId("");
    } catch (error) {
      console.error('Error making team public:', error);
      message.error('Failed to make team public');
    }
  };

  const handleRemovePublicTeam = async (teamId: string) => {
    if (!accessToken) return;

    try {
      await deletePublicTeamCall(accessToken, [teamId]);
      message.success('Successfully removed team from public list');
      
      // Update available teams list
      setAvailableTeams(teams => teams.filter(team => team.team_id !== teamId));
    } catch (error) {
      console.error('Error removing public team:', error);
      message.error('Failed to remove team from public list');
    }
  };

  // Filter out teams that are already public from the dropdown options
  const availableTeamIds = new Set(availableTeams.map(team => team.team_id));
  const privateTeams = allTeams.filter(team => !availableTeamIds.has(team.team_id));

  return (
    <div className="space-y-4">
      <Card className="p-4 flex items-center space-x-4">
        <Select
          value={selectedTeamId}
          onValueChange={setSelectedTeamId}
          placeholder="Select a team to make public"
          className="flex-grow"
        >
          {privateTeams.map((team) => (
            <SelectItem key={team.team_id} value={team.team_id}>
              {team.team_alias || team.team_id}
            </SelectItem>
          ))}
        </Select>
        <Button 
          onClick={handleAddPublicTeam}
          disabled={!selectedTeamId}
          variant="secondary"
        >
          Make Team Public
        </Button>
      </Card>

      <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Team Name</TableHeaderCell>
              <TableHeaderCell>Description</TableHeaderCell>
              <TableHeaderCell>Members</TableHeaderCell>
              <TableHeaderCell>Models</TableHeaderCell>
              <TableHeaderCell>Actions</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {availableTeams.map((team) => (
              <TableRow key={team.team_id}>
                <TableCell>
                  <Text>{team.team_alias}</Text>
                </TableCell>
                <TableCell>
                  <Text>{team.description || 'No description available'}</Text>
                </TableCell>
                <TableCell>
                  <Text>{team.members_with_roles.length} members</Text>
                </TableCell>
                <TableCell>
                  <div className="flex flex-col">
                    {!team.models || team.models.length === 0 ? (
                      <Badge size="xs" color="red">
                        <Text>All Proxy Models</Text>
                      </Badge>
                    ) : (
                      team.models.map((model, index) => (
                        <Badge
                          key={index}
                          size="xs"
                          className="mb-1"
                          color="blue"
                        >
                          <Text>
                            {model.length > 30 ? `${model.slice(0, 30)}...` : model}
                          </Text>
                        </Badge>
                      ))
                    )}
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex space-x-2">
                    <Button 
                      size="xs"
                      variant="secondary" 
                      onClick={() => handleJoinTeam(team.team_id)}
                    >
                      Join Team
                    </Button>
                    <Button 
                      size="xs"
                      variant="light"
                      color="red"
                      onClick={() => handleRemovePublicTeam(team.team_id)}
                    >
                      Remove Public
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
            {availableTeams.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} className="text-center">
                  <Text>No available teams to join</Text>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
};

export default AvailableTeamsPanel;
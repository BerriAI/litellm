import React, { useState, useEffect } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow } from "@tremor/react";
import { Button, Card, Tag } from "antd";
import { availableTeamListCall, teamMemberAddCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

interface AvailableTeam {
  team_id: string;
  team_alias: string;
  description?: string;
  models: string[];
  members_with_roles: { user_id?: string; user_email?: string; role: string }[];
}

interface AvailableTeamsProps {
  accessToken: string | null;
  userID: string | null;
}

const AvailableTeamsPanel: React.FC<AvailableTeamsProps> = ({ accessToken, userID }) => {
  const [availableTeams, setAvailableTeams] = useState<AvailableTeam[]>([]);

  useEffect(() => {
    const fetchAvailableTeams = async () => {
      if (!accessToken || !userID) return;

      try {
        const response = await availableTeamListCall(accessToken);

        setAvailableTeams(response);
      } catch (error) {
        console.error("Error fetching available teams:", error);
      }
    };

    fetchAvailableTeams();
  }, [accessToken, userID]);

  const handleJoinTeam = async (teamId: string) => {
    if (!accessToken || !userID) return;

    try {
      const response = await teamMemberAddCall(accessToken, teamId, {
        user_id: userID,
        role: "user",
      });

      NotificationsManager.success("Successfully joined team");
      // Update available teams list
      setAvailableTeams((teams) => teams.filter((team) => team.team_id !== teamId));
    } catch (error) {
      console.error("Error joining team:", error);
      NotificationsManager.fromBackend("Failed to join team");
    }
  };

  return (
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
              <TableCell>{team.team_alias}</TableCell>
              <TableCell>{team.description || "No description available"}</TableCell>
              <TableCell>{team.members_with_roles.length} members</TableCell>
              <TableCell>
                <div className="flex flex-col items-start">
                  {!team.models || team.models.length === 0 ? (
                    <Tag color="red">All Proxy Models</Tag>
                  ) : (
                    team.models.map((model, index) => (
                      <Tag key={index} color="blue" className="mb-1">
                        {model.length > 30 ? `${model.slice(0, 30)}...` : model}
                      </Tag>
                    ))
                  )}
                </div>
              </TableCell>
              <TableCell>
                <Button size="small" type="primary" ghost onClick={() => handleJoinTeam(team.team_id)}>
                  Join Team
                </Button>
              </TableCell>
            </TableRow>
          ))}
          {availableTeams.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} className="text-center">
                No available teams to join. See how to set available teams{" "}
                <a
                  href="https://docs.litellm.ai/docs/proxy/self_serve#all-settings-for-self-serve--sso-flow"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:text-blue-700 underline"
                >
                  here
                </a>
                .
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </Card>
  );
};

export default AvailableTeamsPanel;

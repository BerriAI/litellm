import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { availableTeamListCall, teamMemberAddCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

interface AvailableTeam {
  team_id: string;
  team_alias: string;
  description?: string;
  models: string[];
  members_with_roles: {
    user_id?: string;
    user_email?: string;
    role: string;
  }[];
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
      await teamMemberAddCall(accessToken, teamId, {
        user_id: userID,
        role: "user",
      });

      NotificationsManager.success("Successfully joined team");
      setAvailableTeams((teams) =>
        teams.filter((team) => team.team_id !== teamId),
      );
    } catch (error) {
      console.error("Error joining team:", error);
      NotificationsManager.fromBackend("Failed to join team");
    }
  };

  return (
    <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Team Name</TableHead>
            <TableHead>Description</TableHead>
            <TableHead>Members</TableHead>
            <TableHead>Models</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {availableTeams.map((team) => (
            <TableRow key={team.team_id}>
              <TableCell>
                <span>{team.team_alias}</span>
              </TableCell>
              <TableCell>
                <span>{team.description || "No description available"}</span>
              </TableCell>
              <TableCell>
                <span>{team.members_with_roles.length} members</span>
              </TableCell>
              <TableCell>
                <div className="flex flex-col">
                  {!team.models || team.models.length === 0 ? (
                    <Badge className="text-xs bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300">
                      All Proxy Models
                    </Badge>
                  ) : (
                    team.models.map((model, index) => (
                      <Badge
                        key={index}
                        className="text-xs mb-1 bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                      >
                        {model.length > 30 ? `${model.slice(0, 30)}...` : model}
                      </Badge>
                    ))
                  )}
                </div>
              </TableCell>
              <TableCell>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => handleJoinTeam(team.team_id)}
                >
                  Join Team
                </Button>
              </TableCell>
            </TableRow>
          ))}
          {availableTeams.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} className="text-center">
                <span>
                  No available teams to join. See how to set available teams{" "}
                  <a
                    href="https://docs.litellm.ai/docs/proxy/self_serve#all-settings-for-self-serve--sso-flow"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:text-primary/80 underline"
                  >
                    here
                  </a>
                  .
                </span>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </Card>
  );
};

export default AvailableTeamsPanel;

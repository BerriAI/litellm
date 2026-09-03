import React, { useState, useEffect } from "react";

import { availableTeamListCall, teamMemberAddCall } from "@/components/networking";
import { toast } from "@/lib/toast";

import AvailableTeamsTable from "./AvailableTeamsTable";
import { AvailableTeam } from "./AvailableTeamsTableColumns";

interface AvailableTeamsProps {
  accessToken: string | null;
  userID: string | null;
}

const AvailableTeamsPanel: React.FC<AvailableTeamsProps> = ({ accessToken, userID }) => {
  const [availableTeams, setAvailableTeams] = useState<AvailableTeam[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let ignore = false;

    const fetchAvailableTeams = async () => {
      if (!accessToken || !userID) {
        setIsLoading(false);
        return;
      }

      try {
        const response = await availableTeamListCall(accessToken);
        if (!ignore) {
          setAvailableTeams(response);
        }
      } catch (error) {
        console.error("Error fetching available teams:", error);
      } finally {
        if (!ignore) {
          setIsLoading(false);
        }
      }
    };

    fetchAvailableTeams();

    return () => {
      ignore = true;
    };
  }, [accessToken, userID]);

  const handleJoinTeam = async (teamId: string) => {
    if (!accessToken || !userID) return;

    try {
      await teamMemberAddCall(accessToken, teamId, {
        user_id: userID,
        role: "user",
      });

      toast.success("Successfully joined team");
      setAvailableTeams((teams) => teams.filter((team) => team.team_id !== teamId));
    } catch (error) {
      console.error("Error joining team:", error);
      toast.fromError("Failed to join team");
    }
  };

  return <AvailableTeamsTable teams={availableTeams} isLoading={isLoading} onJoinTeam={handleJoinTeam} />;
};

export default AvailableTeamsPanel;

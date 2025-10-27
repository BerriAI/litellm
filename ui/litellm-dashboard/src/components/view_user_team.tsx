"use client";
import React, { useEffect, useState } from "react";
import { modelAvailableCall } from "./networking";

interface ViewUserTeamProps {
  userID: string | null;
  userRole: string | null;
  selectedTeam: any | null;
  accessToken: string | null;
}
const ViewUserTeam: React.FC<ViewUserTeamProps> = ({ userID, userRole, selectedTeam, accessToken }) => {
  const [userModels, setUserModels] = useState([]);
  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null) {
          return;
        }

        if (accessToken !== null) {
          const model_available = await modelAvailableCall(accessToken, userID, userRole);
          let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
          console.log("available_model_names:", available_model_names);
          setUserModels(available_model_names);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole]);

  // logic to decide what models to display
  let modelsToDisplay = [];
  if (selectedTeam && selectedTeam.models) {
    modelsToDisplay = selectedTeam.models;
  }

  // check if "all-proxy-models" is in modelsToDisplay
  if (modelsToDisplay && modelsToDisplay.includes("all-proxy-models")) {
    console.log("user models:", userModels);
    modelsToDisplay = userModels;
  }
  return (
    <>
      <div className="mb-5">
        <p className="text-3xl text-tremor-content-strong dark:text-dark-tremor-content-strong font-semibold">
          {selectedTeam?.team_alias}
        </p>
        {selectedTeam?.team_id && (
          <p className="text-xs text-gray-400 dark:text-gray-400 font-semibold">Team ID: {selectedTeam?.team_id}</p>
        )}
      </div>
    </>
  );
};

export default ViewUserTeam;

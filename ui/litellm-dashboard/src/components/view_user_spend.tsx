"use client";
import React, { useEffect, useState } from "react";
import { modelAvailableCall } from "./networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

// Define the props type
interface UserSpendData {
  spend: number; // Adjust the type accordingly based on your data
  max_budget?: number | null; // Optional property with a default of null
  // Add other properties if needed
}
interface ViewUserSpendProps {
  userSpend: number | null;
  userMaxBudget: number | null;
  selectedTeam: any | null;
}
const ViewUserSpend: React.FC<ViewUserSpendProps> = ({ userSpend, userMaxBudget, selectedTeam }) => {
  const { accessToken, userRole, userId: userID } = useAuthorized();
  let [spend, setSpend] = useState(userSpend !== null ? userSpend : 0.0);
  const [maxBudget, setMaxBudget] = useState(
    selectedTeam ? Number(formatNumberWithCommas(selectedTeam.max_budget, 4)) : null,
  );
  useEffect(() => {
    if (selectedTeam) {
      if (selectedTeam.team_alias === "Default Team") {
        setMaxBudget(userMaxBudget);
      } else {
        let setMaxBudgetFlag = false;
        if (selectedTeam.team_memberships) {
          /**
               * What 'team_memberships' looks like:
               * "team_memberships": [
               *  {
               *      "user_id": "2c315de3-e7ce-4269-b73e-b039a06187b1",
               *      "team_id": "test-team_515e6f42-ded2-4f0d-8919-0a1f43c5a45f",
               *      "budget_id": "0880769f-716a-4149-ab19-7f7651ad4db5",
               *      "litellm_budget_table": {
                  "soft_budget": null,
                  "max_budget": 20.0,
                  "max_parallel_requests": null,
                  "tpm_limit": null,
                  "rpm_limit": null,
                  "model_max_budget": null,
                  "budget_duration": null
              }
               */
          for (const member of selectedTeam.team_memberships) {
            if (
              member.user_id === userID &&
              "max_budget" in member.litellm_budget_table &&
              member.litellm_budget_table.max_budget !== null
            ) {
              setMaxBudget(member.litellm_budget_table.max_budget);
              setMaxBudgetFlag = true;
            }
          }
        }
        if (!setMaxBudgetFlag) {
          setMaxBudget(selectedTeam.max_budget);
        }
      }
    } else {
      setMaxBudget(userMaxBudget);
    }
  }, [selectedTeam, userMaxBudget]);
  const [userModels, setUserModels] = useState([]);
  useEffect(() => {
    const fetchData = async () => {
      if (!accessToken || !userID || !userRole) {
        return;
      }
    };
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
    fetchData();
  }, [userRole, accessToken, userID]);

  useEffect(() => {
    if (userSpend !== null) {
      setSpend(userSpend);
    }
  }, [userSpend]);

  // logic to decide what models to display
  let modelsToDisplay = [];
  if (selectedTeam && selectedTeam.models) {
    modelsToDisplay = selectedTeam.models;
  }

  // check if "all-proxy-models" is in modelsToDisplay
  if (modelsToDisplay && modelsToDisplay.includes("all-proxy-models")) {
    console.log("user models:", userModels);
    modelsToDisplay = userModels;
  } else if (modelsToDisplay && modelsToDisplay.includes("all-team-models")) {
    modelsToDisplay = selectedTeam.models;
  } else if (modelsToDisplay && modelsToDisplay.length === 0) {
    modelsToDisplay = userModels;
  }

  const displayMaxBudget = maxBudget !== null ? `$${formatNumberWithCommas(Number(maxBudget), 4)} limit` : "No limit";

  const roundedSpend = spend !== undefined ? formatNumberWithCommas(spend, 4) : null;

  console.log(`spend in view user spend: ${spend}`);
  return (
    <div className="flex items-center">
      <div className="flex justify-between gap-x-6">
        <div>
          <p className="text-tremor-default text-tremor-content dark:text-dark-tremor-content">Total Spend</p>
          <p className="text-2xl text-tremor-content-strong dark:text-dark-tremor-content-strong font-semibold">
            ${roundedSpend}
          </p>
        </div>
        <div>
          <p className="text-tremor-default text-tremor-content dark:text-dark-tremor-content">Max Budget</p>
          <p className="text-2xl text-tremor-content-strong dark:text-dark-tremor-content-strong font-semibold">
            {displayMaxBudget}
          </p>
        </div>
      </div>
      {/* <div className="ml-auto">
          <Accordion>
            <AccordionHeader><Text>Team Models</Text></AccordionHeader>
            <AccordionBody className="absolute right-0 z-10 bg-white p-2 shadow-lg max-w-xs">
              <List>
                {modelsToDisplay.map((model: string) => (
                  <ListItem key={model}>
                    <Text>{model}</Text>
                  </ListItem>
                ))}
              </List>
            </AccordionBody>
          </Accordion>
        </div> */}
    </div>
  );
};

export default ViewUserSpend;

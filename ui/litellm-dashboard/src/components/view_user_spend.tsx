"use client";
import React, { useEffect, useState } from "react";
import { keyDeleteCall, getTotalSpendCall } from "./networking";
import { StatusOnlineIcon, TrashIcon } from "@heroicons/react/outline";
import { Accordion, AccordionHeader, AccordionList, DonutChart } from "@tremor/react";
import {
  Badge,
  Card,
  Table,
  Metric,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
  Title,
  Icon,
  AccordionBody,
  List,
  ListItem,

} from "@tremor/react";
import { Statistic } from "antd"
import { spendUsersCall, modelAvailableCall }  from "./networking";


// Define the props type
interface UserSpendData {
    spend: number; // Adjust the type accordingly based on your data
    max_budget?: number | null; // Optional property with a default of null
    // Add other properties if needed
  }
interface ViewUserSpendProps {
    userID: string | null;
    userRole: string | null;
    accessToken: string | null;
    userSpend: number | null;  
    selectedTeam: any | null;
}
const ViewUserSpend: React.FC<ViewUserSpendProps> = ({ userID, userRole, accessToken, userSpend, selectedTeam }) => {
    console.log(`userSpend: ${userSpend}`)
    let [spend, setSpend] = useState(userSpend !== null ? userSpend : 0.0);
    const [maxBudget, setMaxBudget] = useState(0.0);
    const [userModels, setUserModels] = useState([]);
    useEffect(() => {
      const fetchData = async () => {
        if (!accessToken || !userID || !userRole) {
          return;
        }
        if (userRole === "Admin" && userSpend == null) {
          try {
            const globalSpend = await getTotalSpendCall(accessToken);
            if (globalSpend) {
              if (globalSpend.spend) {
                setSpend(globalSpend.spend);
              } else {
                setSpend(0.0);
              }
              if (globalSpend.max_budget) {
                setMaxBudget(globalSpend.max_budget);
              } else {
                setMaxBudget(0.0);
              }
            }
          } catch (error) {
            console.error("Error fetching global spend data:", error);
          }
        }
      };
      const fetchUserModels = async () => {
        try {
          if (userID === null || userRole === null) {
            return;
          }
  
          if (accessToken !== null) {
            const model_available = await modelAvailableCall(accessToken, userID, userRole);
            let available_model_names = model_available["data"].map(
              (element: { id: string }) => element.id
            );
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
        setSpend(userSpend)
      }
    }, [userSpend])

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


    const displayMaxBudget = maxBudget !== null ? `$${maxBudget} limit` : "No limit";

    const roundedSpend = spend !== undefined ? spend.toFixed(4) : null;

    console.log(`spend in view user spend: ${spend}`)
    return (
      <div className="flex items-center">
        <div>
          <p className="text-tremor-default text-tremor-content dark:text-dark-tremor-content">
            Total Spend{" "}
          </p>
          <p className="text-2xl text-tremor-content-strong dark:text-dark-tremor-content-strong font-semibold">
            ${roundedSpend}
          </p>
        </div>
        <div className="ml-auto">
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
        </div>
      </div>
    );
}

export default ViewUserSpend;


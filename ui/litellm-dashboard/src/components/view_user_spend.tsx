"use client";
import React, { useEffect, useState } from "react";
import { keyDeleteCall, getTotalSpendCall } from "./networking";
import { StatusOnlineIcon, TrashIcon } from "@heroicons/react/outline";
import { DonutChart } from "@tremor/react";
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
} from "@tremor/react";
import { Statistic } from "antd"
import { spendUsersCall }  from "./networking";


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
}
const ViewUserSpend: React.FC<ViewUserSpendProps> = ({ userID, userRole, accessToken, userSpend }) => {
    console.log(`userSpend: ${userSpend}`)
    let [spend, setSpend] = useState(userSpend !== null ? userSpend : 0.0);
    const [maxBudget, setMaxBudget] = useState(0.0);
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
    
      fetchData();
    }, [userRole, accessToken]);

    useEffect(() => {
      if (userSpend !== null) {
        setSpend(userSpend)
      }
    }, [userSpend])

    const displayMaxBudget = maxBudget !== null ? `$${maxBudget} limit` : "No limit";

    const roundedSpend = spend !== undefined ? spend.toFixed(4) : null;

    console.log(`spend in view user spend: ${spend}`)
    return (
        <>
      <p className="text-tremor-default text-tremor-content dark:text-dark-tremor-content">Total Spend </p>
      <p className="text-2xl text-tremor-content-strong dark:text-dark-tremor-content-strong font-semibold">${roundedSpend}</p>
        
    </>
    )
}

export default ViewUserSpend;


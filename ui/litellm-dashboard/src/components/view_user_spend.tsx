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
    userSpendData: UserSpendData | null; // Use the UserSpendData interface here
    userRole: string | null;
    accessToken: string;
}
const ViewUserSpend: React.FC<ViewUserSpendProps> = ({ userID, userSpendData, userRole, accessToken }) => {
    const [spend, setSpend] = useState(userSpendData?.spend);
    const [maxBudget, setMaxBudget] = useState(userSpendData?.max_budget || null);

    useEffect(() => {
      const fetchData = async () => {
        if (userRole === "Admin") {
          try {
            const globalSpend = await getTotalSpendCall(accessToken);
            setSpend(globalSpend.spend);
            setMaxBudget(globalSpend.max_budget || null);
          } catch (error) {
            console.error("Error fetching global spend data:", error);
          }
        }
      };
    
      fetchData();
    }, [userRole, accessToken]);

    const displayMaxBudget = maxBudget !== null ? `$${maxBudget} limit` : "No limit";

    const roundedSpend = spend !== undefined ? spend.toFixed(4) : null;

    return (
        <>
        
        <Statistic title="Total Spend" value={roundedSpend !== null ? roundedSpend : 0} /> / {displayMaxBudget}
    </>
    )
}

export default ViewUserSpend;


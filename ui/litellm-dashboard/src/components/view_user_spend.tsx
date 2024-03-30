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
}
const ViewUserSpend: React.FC<ViewUserSpendProps> = ({ userID, userRole, accessToken }) => {
    const [spend, setSpend] = useState(0.0);
    const [maxBudget, setMaxBudget] = useState(0.0);
    useEffect(() => {
      const fetchData = async () => {
        if (!accessToken || !userID || !userRole) {
          return;
        }
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

    const roundedSpend = spend !== undefined ? spend.toFixed(5) : null;

    return (
        <>
      <p className="text-tremor-default text-tremor-content dark:text-dark-tremor-content">Total Spend</p>
      <p className="text-3xl text-tremor-content-strong dark:text-dark-tremor-content-strong font-semibold">${roundedSpend}</p>
        
       
    </>
    )
}

export default ViewUserSpend;


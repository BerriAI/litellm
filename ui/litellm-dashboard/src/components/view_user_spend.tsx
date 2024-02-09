"use client";
import React, { useEffect, useState } from "react";
import { keyDeleteCall } from "./networking";
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
    console.log("User SpendData:", userSpendData);
    const [spend, setSpend] = useState(userSpendData?.spend);
    const [maxBudget, setMaxBudget] = useState(userSpendData?.max_budget || null);

    useEffect(() => {
        const fetchData = async () => {
            if (userRole === "Admin") {
                try {
                    const data = await spendUsersCall(accessToken, "litellm-proxy-budget");
                    console.log("Result from callSpendUsers:", data);
                    const total_budget = data[0]
                    setSpend(total_budget?.spend);
                    setMaxBudget(total_budget?.max_budget || null);
                } catch (error) {
                    console.error("Failed to get spend for user", error);
                }
            }
        };

        fetchData();
    }, [userRole, accessToken, userID]);

    const displayMaxBudget = maxBudget !== null ? `$${maxBudget} limit` : "No limit";

    return (
        <>
      <Card className="mx-auto mb-4">
        <Metric>
          ${spend}
        </Metric>
        <Title>
            / {displayMaxBudget}
        </Title>
      </Card>
    </>
    )
}

export default ViewUserSpend;


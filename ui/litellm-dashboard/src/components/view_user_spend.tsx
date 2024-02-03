"use client";
import React, { useEffect } from "react";
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


// Define the props type
interface UserSpendData {
    spend: number; // Adjust the type accordingly based on your data
    max_budget?: number | null; // Optional property with a default of null
    // Add other properties if needed
  }
interface ViewUserSpendProps {
    userID: string | null;
    userSpendData: UserSpendData | null; // Use the UserSpendData interface here
}
const ViewUserSpend: React.FC<ViewUserSpendProps> = ({ userID, userSpendData }) => {
    console.log("User SpendData:", userSpendData);
    const spend = userSpendData?.spend;
    const maxBudget = userSpendData?.max_budget || null;
    const displayMaxBudget = maxBudget !== null ? `$${maxBudget} limit` : "No limit";
    const displayText = `$${spend} / ${displayMaxBudget}`;

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


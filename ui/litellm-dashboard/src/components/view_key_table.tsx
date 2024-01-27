"use client";
import React, { useEffect, useState } from "react";
import { userInfoCall } from "./networking";
import { StatusOnlineIcon } from "@heroicons/react/outline";
import {
  Badge,
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
  Title,
} from "@tremor/react";

// Define the props type
interface ViewKeyTableProps {
  userID: string;
  accessToken: string;
  proxyBaseUrl: string;
}

const data = [
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90",
  },
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90",
  },
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90",
  },
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90",
  },
];

const ViewKeyTable: React.FC<ViewKeyTableProps> = ({
  userID,
  accessToken,
  proxyBaseUrl,
}) => {
  const [data, setData] = useState(null); // State to store the data from the API

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await userInfoCall(
          (proxyBaseUrl = proxyBaseUrl),
          (accessToken = accessToken),
          (userID = userID)
        );
        setData(response["keys"]); // Update state with the fetched data
      } catch (error) {
        console.error("There was an error fetching the data", error);
        // Optionally, update your UI to reflect the error state
      }
    };

    fetchData(); // Call the async function to fetch data
  }, []); // Empty dependency array

  if (data == null) {
    return;
  }
  return (
    <Card className="flex-auto overflow-y-auto max-h-[50vh] mb-4">
      <Title>API Keys</Title>
      <Table className="mt-5">
        <TableHead>
          <TableRow>
            <TableHeaderCell>Alias</TableHeaderCell>
            <TableHeaderCell>Secret Key</TableHeaderCell>
            <TableHeaderCell>Spend</TableHeaderCell>
            <TableHeaderCell>Status</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((item) => (
            <TableRow key={item.token}>
              <TableCell>{item.key_alias}</TableCell>
              <TableCell>
                <Text>{item.key_name}</Text>
              </TableCell>
              <TableCell>
                <Text>{item.spend}</Text>
              </TableCell>
              <TableCell>
                <Badge color="emerald" icon={StatusOnlineIcon}>
                  {item.expires}
                </Badge>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  );
};

export default ViewKeyTable;

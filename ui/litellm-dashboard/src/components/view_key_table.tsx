"use client";
import React, { useEffect } from "react";
import { keyDeleteCall } from "./networking";
import { StatusOnlineIcon, TrashIcon } from "@heroicons/react/outline";
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
  Icon,
} from "@tremor/react";

// Define the props type
interface ViewKeyTableProps {
  userID: string;
  accessToken: string;
  proxyBaseUrl: string;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
}

const ViewKeyTable: React.FC<ViewKeyTableProps> = ({
  userID,
  accessToken,
  proxyBaseUrl,
  data,
  setData,
}) => {
  const handleDelete = async (token: String) => {
    if (data == null) {
      return;
    }
    try {
      await keyDeleteCall(proxyBaseUrl, accessToken, token);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      const filteredData = data.filter((item) => item.token !== token);
      setData(filteredData);
    } catch (error) {
      console.error("Error deleting the key:", error);
      // Handle any error situations, such as displaying an error message to the user.
    }
  };

  if (data == null) {
    return;
  }
  console.log("RERENDER TRIGGERED");
  return (
    <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh] mb-4">
      <Title>API Keys</Title>
      <Table className="mt-5">
        <TableHead>
          <TableRow>
            <TableHeaderCell>Secret Key</TableHeaderCell>
            <TableHeaderCell>Spend (USD)</TableHeaderCell>
            <TableHeaderCell>Key Budget (USD)</TableHeaderCell>
            <TableHeaderCell>Expires</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((item) => {
            console.log(item);
            return (
              <TableRow key={item.token}>
                <TableCell>
                  {item.key_name != null ? (
                    <Text>{item.key_name}</Text>
                  ) : (
                    <Text>{item.token}</Text>
                  )
                }
                </TableCell>
                <TableCell>
                  <Text>{item.spend}</Text>
                </TableCell>
                <TableCell>
                  {item.max_budget != null ? (
                    <Text>{item.max_budget}</Text>
                  ) : (
                    <Text>Unlimited Budget</Text>
                  )
                }
                </TableCell>
                <TableCell>
                  {item.expires != null ? (
                    <Text>{item.expires}</Text>
                  ) : (
                    <Text>Never expires</Text>
                  )}
                </TableCell>
                <TableCell>
                  <Icon
                    onClick={() => handleDelete(item.token)}
                    icon={TrashIcon}
                    size="xs"
                  />
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Card>
  );
};

export default ViewKeyTable;

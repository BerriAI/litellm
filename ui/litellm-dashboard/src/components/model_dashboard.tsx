import React, { useState, useEffect } from "react";
import { Card, Title, Table, TableHead, TableRow, TableCell, TableBody } from "@tremor/react";
import { modelInfoCall } from "./networking";

interface ModelDashboardProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

const ModelDashboard: React.FC<ModelDashboardProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const [modelData, setModelData] = useState<any>({ data: [] });

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Replace with your actual API call for model data
        const modelDataResponse = await modelInfoCall(accessToken, token, userRole, userID);

        setModelData(modelDataResponse);
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };

    if (accessToken && token && userRole && userID) {
      fetchData();
    }
  }, [accessToken, token, userRole, userID]);

  return (
    <div style={{ width: "100%" }}>
      <Card>
        <Title>Models Page</Title>
        <Table className="mt-5">
          <TableHead>
            <TableRow>
              <TableCell>Model Name</TableCell>
              <TableCell>Model Info</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {modelData.data.map((model: any) => (
              <TableRow key={model.model_name}>
                <TableCell>{model.model_name}</TableCell>
                {/* <TableCell>{model.model_info}</TableCell> */}
                {/* Add other TableCell for Model Info if needed */}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
};

export default ModelDashboard;

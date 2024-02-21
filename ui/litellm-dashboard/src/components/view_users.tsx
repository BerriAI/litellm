import React, { useState, useEffect } from "react";
import { Card, Title, Subtitle, Table, TableHead, TableRow, TableCell, TableBody, Metric, Grid } from "@tremor/react";
import { userInfoCall } from "./networking";
import { Badge, BadgeDelta, Button } from '@tremor/react';
import RequestAccess from "./request_model_access";
import CreateUser from "./create_user_button";

interface ViewUserDashboardProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

const ViewUserDashboard: React.FC<ViewUserDashboardProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const [userData, setuserData] = useState<null | any[]>(null);
  const [pendingRequests, setPendingRequests] = useState<any[]>([]);


  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    const fetchData = async () => {
      try {
        // Replace with your actual API call for model data
        const userDataResponse = await userInfoCall(accessToken, null,  userRole, true);
        console.log("user data response:", userDataResponse);
        setuserData(userDataResponse);

      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };

    if (accessToken && token && userRole && userID) {
      fetchData();
    }
  }, [accessToken, token, userRole, userID]);

  if (!userData) {
    return <div>Loading...</div>;
  }

  if (!accessToken || !token || !userRole || !userID) {
    return <div>Loading...</div>;
  }

  // when users click request access show pop up to allow them to request access

  return (
    <div style={{ width: "100%" }}>
    <Grid className="gap-2 p-10 h-[75vh] w-full">
        <CreateUser
            userID={userID}
            accessToken={accessToken}
          />
      <Card>
        <Table className="mt-5">
          <TableHead>
            <TableRow>
              <TableCell><Title>User ID </Title></TableCell>
              <TableCell><Title>User Role</Title></TableCell>
              <TableCell><Title>User Models</Title></TableCell>
              <TableCell><Title>User Spend ($ USD)</Title></TableCell>
              <TableCell><Title>User Max Budget  ($ USD)</Title></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {userData.map((user: any) => (
              <TableRow key={user.user_id}>
                <TableCell><Title>{user.user_id}</Title></TableCell>
                <TableCell><Title>{user.user_role ? user.user_role : "app_user"}</Title></TableCell>
                <TableCell><Title>{user.models && user.models.length > 0 ? user.models : "All Models"}</Title></TableCell>
                <TableCell><Title>{user.spend ? user.spend : 0}</Title></TableCell>
                <TableCell><Title>{user.max_budget ? user.max_budget : "Unlimited"}</Title></TableCell>

              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      </Grid>
    </div>
  );
};

export default ViewUserDashboard;

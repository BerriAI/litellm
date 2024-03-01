import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Tab,
  TabGroup,
  TabList,
  Metric,
  Grid,
} from "@tremor/react";
import { userInfoCall } from "./networking";
import { Badge, BadgeDelta, Button } from "@tremor/react";
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
  const [userData, setUserData] = useState<null | any[]>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const defaultPageSize = 25;

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    const fetchData = async () => {
      try {
        // Replace with your actual API call for model data
        const userDataResponse = await userInfoCall(
          accessToken,
          null,
          userRole,
          true
        );
        console.log("user data response:", userDataResponse);
        setUserData(userDataResponse);
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

  function renderPagination() {
    if (!userData) return null;

    const totalPages = Math.ceil(userData.length / defaultPageSize);
    const startItem = (currentPage - 1) * defaultPageSize + 1;
    const endItem = Math.min(currentPage * defaultPageSize, userData.length);

    return (
      <div className="flex justify-between items-center">
        <div>
          Showing {startItem} – {endItem} of {userData.length}
        </div>
        <div className="flex">
          <button
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-l focus:outline-none"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(currentPage - 1)}
          >
            &larr; Prev
          </button>
          <button
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-r focus:outline-none"
            disabled={currentPage === totalPages}
            onClick={() => setCurrentPage(currentPage + 1)}
          >
            Next &rarr;
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: "100%" }}>
      <Grid className="gap-2 p-10 h-[75vh] w-full">
        <CreateUser userID={userID} accessToken={accessToken} />
        <Card>
          <TabGroup>
            <TabList variant="line" defaultValue="1">
              <Tab value="1">Key Owners</Tab>
              <Tab value="2">End-Users</Tab>
            </TabList>
          </TabGroup>
          <Table className="mt-5">
            <TableHead>
              <TableRow>
                <TableCell>
                  <Title>User ID </Title>
                </TableCell>
                <TableCell>
                  <Title>User Role</Title>
                </TableCell>
                <TableCell>
                  <Title>User Models</Title>
                </TableCell>
                <TableCell>
                  <Title>User Spend ($ USD)</Title>
                </TableCell>
                <TableCell>
                  <Title>User Max Budget ($ USD)</Title>
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {userData.map((user: any) => (
                <TableRow key={user.user_id}>
                  <TableCell>{user.user_id}</TableCell>
                  <TableCell>
                    {user.user_role ? user.user_role : "app_owner"}
                  </TableCell>
                  <TableCell>
                    {user.models && user.models.length > 0
                      ? user.models
                      : "All Models"}
                  </TableCell>
                  <TableCell>{user.spend ? user.spend : 0}</TableCell>
                  <TableCell>
                    {user.max_budget ? user.max_budget : "Unlimited"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
        {renderPagination()}
      </Grid>
    </div>
  );
};

export default ViewUserDashboard;

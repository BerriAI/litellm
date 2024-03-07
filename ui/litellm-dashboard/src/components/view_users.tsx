import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableHeaderCell,
  TableRow,
  TableCell,
  TableBody,
  Tab,
  Text,
  TabGroup,
  TabList,
  TabPanels,
  Metric,
  Grid,
  TabPanel,
  Select,
  SelectItem,
} from "@tremor/react";
import { userInfoCall, adminTopEndUsersCall } from "./networking";
import { Badge, BadgeDelta, Button } from "@tremor/react";
import RequestAccess from "./request_model_access";
import CreateUser from "./create_user_button";
import Paragraph from "antd/es/skeleton/Paragraph";

interface ViewUserDashboardProps {
  accessToken: string | null;
  token: string | null;
  keys: any[] | null;
  userRole: string | null;
  userID: string | null;
  setKeys: React.Dispatch<React.SetStateAction<Object[] | null>>;
}

const ViewUserDashboard: React.FC<ViewUserDashboardProps> = ({
  accessToken,
  token,
  keys,
  userRole,
  userID,
  setKeys,
}) => {
  const [userData, setUserData] = useState<null | any[]>(null);
  const [endUsers, setEndUsers] = useState<null | any[]>(null);
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

    if (accessToken && token && userRole && userID && !userData) {
      fetchData();
    }

    const fetchEndUserSpend = async () => {
      try {
        const topEndUsers = await adminTopEndUsersCall(accessToken, null);
        console.log("user data response:", topEndUsers);
        setEndUsers(topEndUsers);
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };
    if (
      userRole &&
      (userRole == "Admin" || userRole == "Admin Viewer") &&
      !endUsers
    ) {
      fetchEndUserSpend();
    }
  }, [accessToken, token, userRole, userID]);

  if (!userData) {
    return <div>Loading...</div>;
  }

  if (!accessToken || !token || !userRole || !userID) {
    return <div>Loading...</div>;
  }

  const onKeyClick = async (keyToken: String) => {
    try {
      const topEndUsers = await adminTopEndUsersCall(accessToken, keyToken);
      console.log("user data response:", topEndUsers);
      setEndUsers(topEndUsers);
    } catch (error) {
      console.error("There was an error fetching the model data", error);
    }
  };

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
        <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh] mb-4">
          <TabGroup>
            <TabList variant="line" defaultValue="1">
              <Tab value="1">Key Owners</Tab>
              <Tab value="2">End-Users</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <Table className="mt-5">
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>User ID</TableHeaderCell>
                      <TableHeaderCell>User Role</TableHeaderCell>
                      <TableHeaderCell>User Models</TableHeaderCell>
                      <TableHeaderCell>User Spend ($ USD)</TableHeaderCell>
                      <TableHeaderCell>User Max Budget ($ USD)</TableHeaderCell>
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
              </TabPanel>
              <TabPanel>
                <div className="flex items-center">
                  <div className="flex-1"></div>
                  <div className="flex-1 flex justify-between items-center">
                    <Text className="w-1/4 mr-2 text-right">Key</Text>
                    <Select defaultValue="1" className="w-3/4">
                      {keys?.map((key: any, index: number) => {
                        if (
                          key &&
                          key["key_name"] !== null &&
                          key["key_name"].length > 0
                        ) {
                          return (
                            <SelectItem
                              key={index}
                              value={String(index)}
                              onClick={() => onKeyClick(key["token"])}
                            >
                              {key["key_name"]}
                            </SelectItem>
                          );
                        }
                      })}
                    </Select>
                  </div>
                </div>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>End User</TableHeaderCell>
                      <TableHeaderCell>Spend</TableHeaderCell>
                      <TableHeaderCell>Total Events</TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    {endUsers?.map((user: any, index: number) => (
                      <TableRow key={index}>
                        <TableCell>{user.end_user}</TableCell>
                        <TableCell>{user.total_spend}</TableCell>
                        <TableCell>{user.total_events}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TabPanel>
            </TabPanels>
          </TabGroup>
        </Card>
        {renderPagination()}
      </Grid>
    </div>
  );
};

export default ViewUserDashboard;

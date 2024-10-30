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
  Dialog,
  DialogPanel,
  Icon,
  TextInput,
} from "@tremor/react";

import { message } from "antd";
import { Modal } from "antd";

import {
  userInfoCall,
  userUpdateUserCall,
  getPossibleUserRoles,
} from "./networking";
import { Badge, BadgeDelta, Button } from "@tremor/react";
import RequestAccess from "./request_model_access";
import CreateUser from "./create_user_button";
import EditUserModal from "./edit_user";
import Paragraph from "antd/es/skeleton/Paragraph";
import {
  PencilAltIcon,
  InformationCircleIcon,
  TrashIcon,
} from "@heroicons/react/outline";

import { userDeleteCall } from "./networking";

interface ViewUserDashboardProps {
  accessToken: string | null;
  token: string | null;
  keys: any[] | null;
  userRole: string | null;
  userID: string | null;
  teams: any[] | null;
  setKeys: React.Dispatch<React.SetStateAction<Object[] | null>>;
}

interface UserListResponse {
  users: any[] | null;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
if (isLocal != true) {
  console.log = function() {};
}

const ViewUserDashboard: React.FC<ViewUserDashboardProps> = ({
  accessToken,
  token,
  keys,
  userRole,
  userID,
  teams,
  setKeys,
}) => {
  const [userListResponse, setUserListResponse] = useState<UserListResponse | null>(null);
  const [userData, setUserData] = useState<null | any[]>(null);
  const [endUsers, setEndUsers] = useState<null | any[]>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [openDialogId, setOpenDialogId] = React.useState<null | number>(null);
  const [selectedItem, setSelectedItem] = useState<null | any>(null);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState<string | null>(null);
  const [possibleUIRoles, setPossibleUIRoles] = useState<
    Record<string, Record<string, string>>
  >({});
  const defaultPageSize = 25;

  const handleDelete = (userId: string) => {
    setUserToDelete(userId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (userToDelete && accessToken) {
      try {
        await userDeleteCall(accessToken, [userToDelete]);
        message.success("User deleted successfully");
        // Update the user list after deletion
        if (userData) {
          const updatedUserData = userData.filter(user => user.user_id !== userToDelete);
          setUserData(updatedUserData);
        }
      } catch (error) {
        console.error("Error deleting user:", error);
        message.error("Failed to delete user");
      }
    }
    setIsDeleteModalOpen(false);
    setUserToDelete(null);
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setUserToDelete(null);
  };

  const handleEditCancel = async () => {
    setSelectedUser(null);
    setEditModalVisible(false);
  };

  const handleEditSubmit = async (editedUser: any) => {
    console.log("inside handleEditSubmit:", editedUser);

    if (!accessToken || !token || !userRole || !userID) {
      return;
    }

    try {
      await userUpdateUserCall(accessToken, editedUser, null);
      message.success(`User ${editedUser.user_id} updated successfully`);
    } catch (error) {
      console.error("There was an error updating the user", error);
    }
    if (userData) {
      const updatedUserData = userData.map((user) =>
        user.user_id === editedUser.user_id ? editedUser : user
      );
      setUserData(updatedUserData);
    }
    setSelectedUser(null);
    setEditModalVisible(false);
    // Close the modal
  };

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
          true,
          currentPage,
          defaultPageSize
        );

        setUserListResponse(userDataResponse);

        console.log("user data response:", userDataResponse);
        setUserData(userDataResponse.users || []);

        const availableUserRoles = await getPossibleUserRoles(accessToken);
        setPossibleUIRoles(availableUserRoles);
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };

    if (accessToken && token && userRole && userID) {
      fetchData();
    }
  }, [accessToken, token, userRole, userID, currentPage]);

  if (!userData) {
    return <div>Loading...</div>;
  }

  if (!accessToken || !token || !userRole || !userID) {
    return <div>Loading...</div>;
  }

  function renderPagination() {
    if (!userData) return null;

    const totalPages = userListResponse?.total_pages || 0;

    const handlePageChange = (newPage: number) => {
      setUserData([]); // Clear current users
      setCurrentPage(newPage);
    };
  

    return (
      <div className="flex justify-between items-center">
        <div>
          Showing Page {currentPage } of {totalPages}
        </div>
        <div className="flex">
          <button
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-l focus:outline-none"
            disabled={currentPage === 1}
            onClick={() => handlePageChange(currentPage - 1)}
          >
            &larr; Prev
          </button>
          <button
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-r focus:outline-none"
            disabled={currentPage === totalPages}
            onClick={() => handlePageChange(currentPage + 1)}
          >
            Next &rarr;
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: "100%" }}>
      <Grid className="gap-2 p-2 h-[90vh] w-full mt-8">
        <CreateUser
          userID={userID}
          accessToken={accessToken}
          teams={teams}
          possibleUIRoles={possibleUIRoles}
        />
        <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[90vh] mb-4">
          <div className="mb-4 mt-1"></div>
          <TabGroup>
            <TabPanels>
              <TabPanel>
                <Table className="mt-5">
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>User ID</TableHeaderCell>
                      <TableHeaderCell>User Email</TableHeaderCell>
                      <TableHeaderCell>Role</TableHeaderCell>
                      <TableHeaderCell>User Spend ($ USD)</TableHeaderCell>
                      <TableHeaderCell>User Max Budget ($ USD)</TableHeaderCell>
                      <TableHeaderCell>API Keys</TableHeaderCell>
                      <TableHeaderCell></TableHeaderCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {userData.map((user: any) => (
                      <TableRow key={user.user_id}>
                        <TableCell>{user.user_id || "-"}</TableCell>
                        <TableCell>{user.user_email || "-"}</TableCell>
                        <TableCell>
                          {possibleUIRoles?.[user?.user_role]?.ui_label || "-"}
                        </TableCell>
                        <TableCell>
                          {user.spend ? user.spend?.toFixed(2) : "-"}
                        </TableCell>
                        <TableCell>
                          {user.max_budget !== null ? user.max_budget : "Unlimited"}
                        </TableCell>
                        <TableCell>
                          <Grid numItems={2}>
                            {user.key_count > 0 ? (
                              <Badge size={"xs"} color={"indigo"}>
                                {user.key_count} Keys
                              </Badge>
                            ) : (
                              <Badge size={"xs"} color={"gray"}>
                                No Keys
                              </Badge>
                            )}
                            {/* <Text>{user.key_aliases.filter(key => key !== null).length} Keys</Text> */}
                          </Grid>
                        </TableCell>
                        <TableCell>
                          <Icon
                            icon={PencilAltIcon}
                            onClick={() => {
                              setSelectedUser(user);
                              setEditModalVisible(true);
                            }}
                          >
                            View Keys
                          </Icon>
                          <Icon
                            icon={TrashIcon}
                            onClick={() => handleDelete(user.user_id)}
                          >
                            Delete
                          </Icon>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TabPanel>
              <TabPanel>
                <div className="flex items-center">
                  <div className="flex-1"></div>
                  <div className="flex-1 flex justify-between items-center"></div>
                </div>
              </TabPanel>
            </TabPanels>
          </TabGroup>
          <EditUserModal
            visible={editModalVisible}
            possibleUIRoles={possibleUIRoles}
            onCancel={handleEditCancel}
            user={selectedUser}
            onSubmit={handleEditSubmit}
          />
          {isDeleteModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div
              className="fixed inset-0 transition-opacity"
              aria-hidden="true"
            >
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            {/* Modal Panel */}
            <span
              className="hidden sm:inline-block sm:align-middle sm:h-screen"
              aria-hidden="true"
            >
              &#8203;
            </span>

            {/* Confirmation Modal Content */}
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Delete User
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to delete this user?
                      </p>
                      <p className="text-sm font-medium text-gray-900 mt-2">
                        User ID: {userToDelete}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button onClick={confirmDelete} color="red" className="ml-2">
                  Delete
                </Button>
                <Button onClick={cancelDelete}>Cancel</Button>
              </div>
            </div>
          </div>
        </div>
      )}
        </Card>
        {renderPagination()}
      </Grid>
    </div>
  );
};

export default ViewUserDashboard;

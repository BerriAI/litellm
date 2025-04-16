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
import { columns } from "./view_users/columns";
import { UserDataTable } from "./view_users/table";
import { UserInfo } from "./view_users/types";
import BulkCreateUsers from "./bulk_create_users_button";
import SSOSettings from "./SSOSettings";

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

interface CreateuserProps {
  userID: string;
  accessToken: string;
  teams: any[];
  possibleUIRoles: Record<string, Record<string, string>>;
  onUserCreated: () => Promise<void>;
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
  const [selectedUser, setSelectedUser] = useState<UserInfo | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState<string | null>(null);
  const [possibleUIRoles, setPossibleUIRoles] = useState<
    Record<string, Record<string, string>>
  >({});
  const defaultPageSize = 25;
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTab, setActiveTab] = useState("users");

  // check if window is not undefined
  if (typeof window !== "undefined") {
    window.addEventListener("beforeunload", function () {
      // Clear session storage
      sessionStorage.clear();
    });
  }

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

  const refreshUserData = async () => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    
    try {
      const userDataResponse = await userInfoCall(
        accessToken,
        null,
        userRole,
        true,
        currentPage,
        defaultPageSize
      );
      
      // Update session storage with new data
      sessionStorage.setItem(
        `userList_${currentPage}`,
        JSON.stringify(userDataResponse)
      );
      
      setUserListResponse(userDataResponse);
      setUserData(userDataResponse.users || []);
    } catch (error) {
      console.error("Error refreshing user data:", error);
    }
  };

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    const fetchData = async () => {
      try {
        // Check session storage first
        const cachedUserData = sessionStorage.getItem(`userList_${currentPage}`);
        if (cachedUserData) {
          const parsedData = JSON.parse(cachedUserData);
          setUserListResponse(parsedData);
          setUserData(parsedData.users || []);
        } else {
          // Fetch from API if not in cache
          const userDataResponse = await userInfoCall(
            accessToken,
            null,
            userRole,
            true,
            currentPage,
            defaultPageSize
          );

          // Store in session storage
          sessionStorage.setItem(
            `userList_${currentPage}`,
            JSON.stringify(userDataResponse)
          );

          setUserListResponse(userDataResponse);
          setUserData(userDataResponse.users || []);
        }

        // Fetch roles if not cached
        const cachedRoles = sessionStorage.getItem('possibleUserRoles');
        if (cachedRoles) {
          setPossibleUIRoles(JSON.parse(cachedRoles));
        } else {
          const availableUserRoles = await getPossibleUserRoles(accessToken);
          sessionStorage.setItem('possibleUserRoles', JSON.stringify(availableUserRoles));
          setPossibleUIRoles(availableUserRoles);
        }
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

  const tableColumns = columns(
    possibleUIRoles,
    (user) => {
      setSelectedUser(user);
      setEditModalVisible(true);
    },
    handleDelete
  );

  return (
    <div className="w-full p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Users</h1>
        <div className="flex space-x-3">
          <CreateUser
            userID={userID}
            accessToken={accessToken}
            teams={teams}
            possibleUIRoles={possibleUIRoles}
          />
        </div>
      </div>
      
      <TabGroup defaultIndex={0} onIndexChange={(index) => setActiveTab(index === 0 ? "users" : "settings")}>
        <TabList className="mb-4">
          <Tab>Users</Tab>
          <Tab>Default User Settings</Tab>
        </TabList>
        
        <TabPanels>
          <TabPanel>
            <div className="bg-white rounded-lg shadow">
              <div className="border-b px-6 py-4">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-gray-700">
                      Showing{" "}
                      {userListResponse && userListResponse.users && userListResponse.users.length > 0
                        ? (userListResponse.page - 1) * userListResponse.page_size + 1
                        : 0}{" "}
                      -{" "}
                      {userListResponse && userListResponse.users
                        ? Math.min(
                            userListResponse.page * userListResponse.page_size,
                            userListResponse.total
                          )
                        : 0}{" "}
                      of {userListResponse ? userListResponse.total : 0} results
                    </span>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                        disabled={!userListResponse || currentPage <= 1}
                        className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Previous
                      </button>
                      <span className="text-sm text-gray-700">
                        Page {userListResponse ? userListResponse.page : "-"} of{" "}
                        {userListResponse ? userListResponse.total_pages : "-"}
                      </span>
                      <button
                        onClick={() => setCurrentPage((p) => p + 1)}
                        disabled={
                          !userListResponse ||
                          currentPage >= userListResponse.total_pages
                        }
                        className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              <UserDataTable
                data={userData || []}
                columns={tableColumns}
                isLoading={!userData}
              />
            </div>
          </TabPanel>
          
          <TabPanel>
            <SSOSettings accessToken={accessToken} possibleUIRoles={possibleUIRoles} userID={userID} userRole={userRole}/>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      {/* Existing Modals */}
      <EditUserModal
        visible={editModalVisible}
        possibleUIRoles={possibleUIRoles}
        onCancel={handleEditCancel}
        user={selectedUser}
        onSubmit={handleEditSubmit}
      />

      {/* Delete Confirmation Modal */}
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
    </div>
  );
};

export default ViewUserDashboard;

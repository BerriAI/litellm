import React, { useState, useEffect, useRef, Fragment } from "react";
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
  Button,
} from "@tremor/react";

import { message, Modal } from "antd";

import {
  userInfoCall,
  userListCall,
  userUpdateUserCall,
  getPossibleUserRoles,
  userDeleteCall,
} from "./networking";

import { Badge, BadgeDelta } from "@tremor/react";
import RequestAccess from "./request_model_access";
import CreateUser from "./create_user_button";
import EditUserModal from "./edit_user";
import Paragraph from "antd/es/skeleton/Paragraph";
import {
  PencilAltIcon,
  InformationCircleIcon,
  TrashIcon,
} from "@heroicons/react/outline";

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
if (isLocal !== true) {
  console.log = function () {};
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
  const [searchTerm, setSearchTerm] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [keyNameFilter, setKeyNameFilter] = useState("");
  const [teamNameFilter, setTeamNameFilter] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setShowFilters(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Clear session storage on unload
  useEffect(() => {
    if (typeof window !== "undefined") {
      window.addEventListener("beforeunload", function () {
        sessionStorage.clear();
      });
    }
  }, []);

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

  const handleEditCancel = () => {
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
      message.error("Failed to update user");
    }
    if (userData) {
      const updatedUserData = userData.map((user) =>
        user.user_id === editedUser.user_id ? editedUser : user
      );
      setUserData(updatedUserData);
    }
    setSelectedUser(null);
    setEditModalVisible(false);
  };

  const fetchData = async () => {
    try {
      // Fetch from API with search parameters
      const userDataResponse = await userListCall(
        accessToken,
        currentPage,
        defaultPageSize,
        searchTerm,
        keyNameFilter,
        teamNameFilter
      );

      // Store in session storage
      sessionStorage.setItem(
        `userList_${currentPage}`,
        JSON.stringify(userDataResponse)
      );

      setUserListResponse(userDataResponse);
      setUserData(userDataResponse.users || []);

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
      message.error("Failed to fetch user data");
    }
  };

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    fetchData();
  }, [accessToken, token, userRole, userID, currentPage, searchTerm, keyNameFilter, teamNameFilter]);

  if (!userData) {
    return <div>Loading...</div>;
  }

  if (!accessToken || !token || !userRole || !userID) {
    return <div>Loading...</div>;
  }

  const handleSearch = () => {
    setCurrentPage(1);
    fetchData();
  };

  return (
    <div className="w-full">
      <h1 className="text-xl font-semibold mb-4">Users</h1>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="border-b px-6 py-4">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative" ref={dropdownRef}>
                <button
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                  onClick={() => setShowFilters(!showFilters)}
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                    />
                  </svg>
                  Filters
                </button>

                {showFilters && (
                  <div className="absolute left-0 mt-2 w-[500px] bg-white rounded-lg shadow-lg border p-4 z-50">
                    <div className="flex flex-col space-y-4">
                      <TextInput
                        placeholder="Search by User ID or Email"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                      />
                      <div className="flex space-x-2">
                        <TextInput
                          placeholder="Filter by Key Name"
                          value={keyNameFilter}
                          onChange={(e) => setKeyNameFilter(e.target.value)}
                        />
                        <TextInput
                          placeholder="Filter by Team Name"
                          value={teamNameFilter}
                          onChange={(e) => setTeamNameFilter(e.target.value)}
                        />
                      </div>
                      <div className="flex justify-end space-x-2">
                        <Button onClick={handleSearch}>Apply</Button>
                        <Button
                          variant="secondary"
                          onClick={() => {
                            setSearchTerm("");
                            setKeyNameFilter("");
                            setTeamNameFilter("");
                            setShowFilters(false);
                          }}
                        >
                          Reset
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <CreateUser
                userID={userID}
                accessToken={accessToken}
                teams={teams}
                possibleUIRoles={possibleUIRoles}
              />
            </div>

            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">
                Showing {((currentPage - 1) * defaultPageSize) + 1} -{" "}
                {Math.min(currentPage * defaultPageSize, userListResponse?.total || 0)} of{" "}
                {userListResponse?.total || 0} results
              </span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-700">
                  Page {currentPage} of {userListResponse?.total_pages || 1}
                </span>
                <button
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => setCurrentPage((p) => Math.min(userListResponse?.total_pages || 1, p + 1))}
                  disabled={currentPage === (userListResponse?.total_pages || 1)}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>

        <Card className="w-full mx-auto">
          <Table className="mt-5">
            <TableHead>
              <TableRow>
                <TableHeaderCell>User ID</TableHeaderCell>
                <TableHeaderCell>User Email</TableHeaderCell>
                <TableHeaderCell>Role</TableHeaderCell>
                <TableHeaderCell>User Spend ($ USD)</TableHeaderCell>
                <TableHeaderCell>User Max Budget ($ USD)</TableHeaderCell>
                <TableHeaderCell>API Keys</TableHeaderCell>
                <TableHeaderCell>Actions</TableHeaderCell>
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
                    </Grid>
                  </TableCell>
                  <TableCell>
                    <Icon
                      icon={PencilAltIcon}
                      onClick={() => {
                        setSelectedUser(user);
                        setEditModalVisible(true);
                      }}
                      className="cursor-pointer mr-2"
                    />
                    <Icon
                      icon={TrashIcon}
                      onClick={() => handleDelete(user.user_id)}
                      className="cursor-pointer"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
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
      </div>
    </div>
  );
};

export default ViewUserDashboard;

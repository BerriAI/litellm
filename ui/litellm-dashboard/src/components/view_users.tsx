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
  const [filterType, setFilterType] = useState<'user_id' | 'user_email' | ''>('');
  const [filterValue, setFilterValue] = useState('');

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

  const getFilterParams = (filterType: string, searchValue: string) => {
    return {
      user_id: filterType === 'user_id' ? searchValue : null,
      user_email: filterType === 'user_email' ? searchValue : null
    };
  };

  const fetchData = async () => {
    try {
      const { user_id, user_email } = getFilterParams(filterType, searchTerm);
      const userDataResponse = await userListCall(
        accessToken,
        currentPage,
        defaultPageSize,
        user_id,
        user_email
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
  }, [accessToken, token, userRole, userID, currentPage, searchTerm, keyNameFilter, teamNameFilter, filterType]);

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
    <div className="p-4">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <svg
              className="-ml-0.5 mr-2 h-4 w-4"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
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
          <CreateUser
            userID={userID}
            accessToken={accessToken}
            teams={teams}
            possibleUIRoles={possibleUIRoles}
          />
        </div>

        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <span>
            Showing {((currentPage - 1) * defaultPageSize) + 1} - {Math.min(currentPage * defaultPageSize, userListResponse?.total || 0)} of {userListResponse?.total || 0} results
          </span>
          <div className="flex items-center space-x-2">
            <span>Page {currentPage} of {userListResponse?.total_pages || 1}</span>
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(userListResponse?.total_pages || 1, p + 1))}
              disabled={currentPage === (userListResponse?.total_pages || 1)}
              className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {showFilters && (
        <div className="mb-6 bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">Where</div>
            
            <Select 
              value={filterType}
              onValueChange={(value) => setFilterType(value as 'user_id' | 'user_email')}
              className="min-w-[200px]"
            >
              <SelectItem value="">Select filter type...</SelectItem>
              <SelectItem value="user_id">User ID</SelectItem>
              <SelectItem value="user_email">User Email</SelectItem>
            </Select>
            
            <div className="text-sm text-gray-500">Equals</div>
            
            <TextInput
              placeholder="Enter value..."
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              className="flex-1 max-w-md"
            />

            <Button
              onClick={() => {
                if (filterType && filterValue) {
                  setSearchTerm(filterValue);
                  setCurrentPage(1);
                  fetchData();
                }
              }}
              size="sm"
              variant="primary"
              disabled={!filterType || !filterValue}
              className="bg-indigo-500 hover:bg-indigo-600"
            >
              Apply Filter
            </Button>
            
            <Button
              onClick={() => {
                setFilterType('');
                setFilterValue('');
                setSearchTerm('');
                setShowFilters(false);
                setCurrentPage(1);
                fetchData();
              }}
              size="sm"
              variant="secondary"
              color="gray"
              className="border border-gray-300"
            >
              Clear
            </Button>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <Table>
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
                  <TableCell className="max-w-[200px] truncate">
                    {user.user_id || "-"}
                  </TableCell>
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
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${user.key_count > 0 ? 'bg-indigo-100 text-indigo-800' : 'bg-gray-100 text-gray-800'}`}>
                      {user.key_count > 0 ? `${user.key_count} Keys` : "No Keys"}
                    </span>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => {
                          setSelectedUser(user);
                          setEditModalVisible(true);
                        }}
                        className="text-gray-600 hover:text-gray-900"
                      >
                        <PencilAltIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => handleDelete(user.user_id)}
                        className="text-gray-600 hover:text-gray-900"
                      >
                        <TrashIcon className="h-5 w-5" />
                      </button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>

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
    </div>
  );
};

export default ViewUserDashboard;


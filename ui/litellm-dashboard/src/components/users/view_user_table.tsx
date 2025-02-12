import React, { useState, useEffect } from 'react';
import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
  Badge,
  Icon,
  TextInput,
  Select,
  SelectItem,
  Button,
} from "@tremor/react";
import { PencilAltIcon, TrashIcon, SearchIcon } from "@heroicons/react/outline";

import { userFilterUICall } from '../networking';

interface User {
  user_id: string;
  user_email: string | null;
  user_role: string;
  spend: number | null;
  max_budget: number | null;
  key_count: number;
}

interface UserListResponse {
  users: User[] | null;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface PossibleRole {
  ui_label: string;
  [key: string]: string;
}

interface FilterableUserTableProps {
  accessToken: string;
  possibleUIRoles: Record<string, PossibleRole>;
  onEdit: (user: User) => void;
  onDelete: (userId: string) => void;
  defaultPageSize?: number;
}

const FilterableUserTable: React.FC<FilterableUserTableProps> = ({ 
  accessToken,
  possibleUIRoles,
  onEdit,
  onDelete,
  defaultPageSize = 25
}) => {
  const [searchType, setSearchType] = useState<'email' | 'user_id'>('email');
  const [searchValue, setSearchValue] = useState<string>('');
  const [users, setUsers] = useState<User[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUsers = async (
    page: number,
    searchValue: string,
    searchType: 'email' | 'user_id'
  ) => {
    setIsLoading(true);
    setError(null);
    const params = new URLSearchParams();
    
    if (searchValue.trim()) {
      if (searchType === 'email') {
        params.append("user_email", searchValue.trim());
      } else {
        params.append("user_id", searchValue.trim());
      }
    }
    params.append("page", page.toString());
    params.append("page_size", defaultPageSize.toString());
    
    if (!accessToken) {
      return;
    }

    try {
      const response = await userFilterUICall(accessToken, params);
      setUsers(response.users || []);
      setTotalPages(response.total_pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch users');
      console.error('Error fetching users:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = () => {
    setCurrentPage(1);
    fetchUsers(1, searchValue, searchType);
  };

  useEffect(() => {
    if (!accessToken) return;
    fetchUsers(1, '', 'email');
  }, [accessToken]);

  useEffect(() => {
    if (currentPage > 1) {
      fetchUsers(currentPage, searchValue, searchType);
    }
  }, [currentPage]);



  return (
    <div className="space-y-4">
      <div className="flex gap-2 items-center">
        <Select
          value={searchType}
          onValueChange={(value: 'email' | 'user_id') => setSearchType(value)}
          className="w-32"
        >
          <SelectItem value="email">Email</SelectItem>
          <SelectItem value="user_id">User ID</SelectItem>
        </Select>
        <div className="relative flex-1">
          <TextInput
            placeholder={`Search by ${searchType === 'email' ? 'email' : 'user ID'}...`}
            value={searchValue}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchValue(e.target.value)}
            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
              if (e.key === 'Enter') {
                handleSearch();
              }
            }}
            icon={SearchIcon}
            className="w-full"
          />
        </div>
        <Button
          onClick={handleSearch}
          size="md"
          className="px-6"
        >
          Search
        </Button>
      </div>

      {isLoading && (
        <div className="text-center py-4">Loading...</div>
      )}

      {error && (
        <div className="text-center py-4 text-red-500">
          Error: {error}
        </div>
      )}

      <Table>
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
          {users.map((user) => (
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
                <Badge 
                  size="xs" 
                  color={user.key_count > 0 ? "indigo" : "gray"}
                >
                  {user.key_count > 0 ? `${user.key_count} Keys` : "No Keys"}
                </Badge>
              </TableCell>
              <TableCell>
                <div className="flex gap-2">
                  <Icon
                    icon={PencilAltIcon}
                    onClick={() => onEdit(user)}
                    className="cursor-pointer"
                  />
                  <Icon
                    icon={TrashIcon}
                    onClick={() => onDelete(user.user_id)}
                    className="cursor-pointer"
                  />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      
      {users.length === 0 && !isLoading && (
        <div className="text-center py-4 text-gray-500">
          No users found matching your search criteria
        </div>
      )}

      <div className="flex justify-between items-center mt-4">
        <div className="text-gray-600">
          Showing Page {currentPage} of {totalPages}
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={currentPage === 1 || isLoading}
            variant="secondary"
            size="md"
            className="px-6"
          >
            Previous
          </Button>
          <Button
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={currentPage === totalPages || isLoading}
            variant="secondary"
            size="md"
            className="px-6"
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
};

export default FilterableUserTable;
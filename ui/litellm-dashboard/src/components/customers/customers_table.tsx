import React from "react";
import { Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell, Text, Badge } from "@tremor/react";

interface CustomerSpendData {
  end_user_id: string;
  alias: string | null;
  total_spend: number;
  total_requests: number;
  total_tokens: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
}

interface CustomersTableProps {
  customers: CustomerSpendData[];
  loading: boolean;
  page: number;
  pageSize: number;
  totalPages: number;
  totalCustomers: number;
  onPageChange: (page: number) => void;
  onCustomerClick: (customerId: string) => void;
}

const CustomersTable: React.FC<CustomersTableProps> = ({
  customers,
  loading,
  page,
  pageSize,
  totalPages,
  totalCustomers,
  onPageChange,
  onCustomerClick,
}) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat("en-US").format(value);
  };

  return (
    <div className="flex flex-col">
      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>End User ID</TableHeaderCell>
            <TableHeaderCell>Alias</TableHeaderCell>
            <TableHeaderCell className="text-right">Total Spend</TableHeaderCell>
            <TableHeaderCell className="text-right">Requests</TableHeaderCell>
            <TableHeaderCell className="text-right">Total Tokens</TableHeaderCell>
            <TableHeaderCell className="text-right">Prompt Tokens</TableHeaderCell>
            <TableHeaderCell className="text-right">Completion Tokens</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={7} className="text-center py-8">
                <Text className="text-gray-500">Loading...</Text>
              </TableCell>
            </TableRow>
          ) : customers.length > 0 ? (
            customers.map((customer) => (
              <TableRow
                key={customer.end_user_id}
                className="hover:bg-gray-50 cursor-pointer"
                onClick={() => onCustomerClick(customer.end_user_id)}
              >
                <TableCell>
                  <Text className="font-medium text-blue-600 hover:text-blue-800">
                    {customer.end_user_id}
                  </Text>
                </TableCell>
                <TableCell>
                  {customer.alias ? (
                    <Badge color="blue">{customer.alias}</Badge>
                  ) : (
                    <Text className="text-gray-400">-</Text>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  <Text className="font-semibold">{formatCurrency(customer.total_spend)}</Text>
                </TableCell>
                <TableCell className="text-right">
                  <Text>{formatNumber(customer.total_requests)}</Text>
                </TableCell>
                <TableCell className="text-right">
                  <Text>{formatNumber(customer.total_tokens)}</Text>
                </TableCell>
                <TableCell className="text-right">
                  <Text>{formatNumber(customer.total_prompt_tokens)}</Text>
                </TableCell>
                <TableCell className="text-right">
                  <Text>{formatNumber(customer.total_completion_tokens)}</Text>
                </TableCell>
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={7} className="text-center py-8">
                <Text className="text-gray-500">No customers found</Text>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      <div className="flex items-center justify-between px-6 py-4 border-t">
        <Text className="text-sm text-gray-600">
          Showing {customers.length > 0 ? (page - 1) * pageSize + 1 : 0} to{" "}
          {Math.min(page * pageSize, totalCustomers)} of {totalCustomers} customers
        </Text>
        {totalPages > 1 && (
          <div className="flex items-center gap-4">
            <button
              onClick={() => onPageChange(page - 1)}
              disabled={page === 1}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <Text className="text-sm text-gray-700">
              Page {page} of {totalPages}
            </Text>
            <button
              onClick={() => onPageChange(page + 1)}
              disabled={page === totalPages}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default CustomersTable;

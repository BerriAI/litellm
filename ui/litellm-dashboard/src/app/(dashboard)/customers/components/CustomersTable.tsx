import React from "react";
import {
  Button,
  Icon,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Badge,
  Text,
} from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { Customer } from "@/app/(dashboard)/customers/types";

interface CustomersTableProps {
  customers: Customer[];
  userRole: string | null;
  onEdit: (customer: Customer) => void;
  onDelete: (customer: Customer) => void;
  onViewInfo: (customer: Customer) => void;
  isLoading: boolean;
}

const CustomersTable: React.FC<CustomersTableProps> = ({
  customers,
  userRole,
  onEdit,
  onDelete,
  onViewInfo,
  isLoading,
}) => {
  const truncateId = (id: string) => {
    if (id.length <= 10) return id;
    return id.substring(0, 7) + "...";
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Text>Loading customers...</Text>
      </div>
    );
  }

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeaderCell>Customer Name</TableHeaderCell>
          <TableHeaderCell>Customer ID</TableHeaderCell>
          <TableHeaderCell>Spend (USD)</TableHeaderCell>
          <TableHeaderCell>Budget (USD)</TableHeaderCell>
          <TableHeaderCell>Default Model</TableHeaderCell>
          <TableHeaderCell>Region</TableHeaderCell>
          <TableHeaderCell>Status</TableHeaderCell>
          {(userRole === "Admin" || userRole === "Org Admin") && (
            <TableHeaderCell>Actions</TableHeaderCell>
          )}
        </TableRow>
      </TableHead>
      <TableBody>
        {customers.length === 0 ? (
          <TableRow>
            <TableCell colSpan={8} className="text-center py-12">
              <Text>No customers found.</Text>
            </TableCell>
          </TableRow>
        ) : (
          customers.map((customer) => (
            <TableRow key={customer.user_id}>
              <TableCell>{customer.alias || "—"}</TableCell>
              <TableCell>
                <Tooltip title={customer.user_id}>
                  <Button
                    size="xs"
                    variant="light"
                    className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100"
                    onClick={() => onViewInfo(customer)}
                  >
                    {truncateId(customer.user_id)}
                  </Button>
                </Tooltip>
              </TableCell>
              <TableCell>{formatNumberWithCommas(customer.spend, 4)}</TableCell>
              <TableCell>
                {customer.litellm_budget_table?.max_budget !== null &&
                customer.litellm_budget_table?.max_budget !== undefined
                  ? customer.litellm_budget_table.max_budget.toString()
                  : "No limit"}
              </TableCell>
              <TableCell>
                {customer.default_model ? (
                  <Badge color="gray">{customer.default_model}</Badge>
                ) : (
                  <Text>—</Text>
                )}
              </TableCell>
              <TableCell>
                {customer.allowed_model_region ? (
                  <Text>{customer.allowed_model_region.toUpperCase()}</Text>
                ) : (
                  <Text>—</Text>
                )}
              </TableCell>
              <TableCell>
                {customer.blocked ? (
                  <Badge color="red">Blocked</Badge>
                ) : (
                  <Badge color="green">Active</Badge>
                )}
              </TableCell>
              {(userRole === "Admin" || userRole === "Org Admin") && (
                <TableCell>
                  <div className="flex items-center gap-1">
                    <Icon
                      icon={PencilAltIcon}
                      size="sm"
                      onClick={() => onEdit(customer)}
                      className="cursor-pointer"
                    />
                    <Icon
                      icon={TrashIcon}
                      size="sm"
                      onClick={() => onDelete(customer)}
                      className="cursor-pointer"
                    />
                  </div>
                </TableCell>
              )}
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
};

export default CustomersTable;

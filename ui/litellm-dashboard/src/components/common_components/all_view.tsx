import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
  Badge,
  Icon,
  Button,
  Card,
} from "@tremor/react";
import { Tooltip } from "antd";

interface Column {
  header: string;
  accessor: string;
  cellRenderer?: (value: any, row: any) => React.ReactNode;
  width?: string;
  style?: React.CSSProperties;
}

interface Action<T = any> {
    icon?: React.ComponentType<any>;
    onClick: (item: T) => void;
    condition?: () => boolean;
    tooltip?: string;
}

interface DeleteModalProps {
  isOpen: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  title: string;
  message: string;
}

interface DataTableProps {
  data: any[];
  columns: Column[];
  actions?: Action[];
  emptyMessage?: string;
  deleteModal?: DeleteModalProps;
}

const DataTable: React.FC<DataTableProps> = ({
  data,
  columns,
  actions,
  emptyMessage = "No data available",
  deleteModal
}) => {
  const renderCell = (column: Column, row: any) => {
    const value = row[column.accessor];
    
    if (column.cellRenderer) {
      return column.cellRenderer(value, row);
    }

    // Default cell rendering based on value type
    if (Array.isArray(value)) {
      return (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {value.length === 0 ? (
            <Badge size="xs" className="mb-1" color="red">
              <Text>None</Text>
            </Badge>
          ) : (
            value.map((item: any, index: number) => (
              <Badge
                key={index}
                size="xs"
                className="mb-1"
                color="blue"
              >
                <Text>
                  {String(item).length > 30
                    ? `${String(item).slice(0, 30)}...`
                    : item}
                </Text>
              </Badge>
            ))
          )}
        </div>
      );
    }

    return value?.toString() || '';
  };

  return (
    <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
      <Table>
        <TableHead>
          <TableRow>
            {columns.map((column, index) => (
              <TableHeaderCell key={index}>{column.header}</TableHeaderCell>
            ))}
            {actions && actions.length > 0 && (
              <TableHeaderCell>Actions</TableHeaderCell>
            )}
          </TableRow>
        </TableHead>

        <TableBody>
          {data && data.length > 0 ? (
            data.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                {columns.map((column, colIndex) => (
                  <TableCell
                    key={colIndex}
                    style={{
                      maxWidth: column.width || "4px",
                      whiteSpace: "pre-wrap",
                      overflow: "hidden",
                      ...column.style
                    }}
                  >
                    {column.accessor === 'id' ? (
                      <Tooltip title={row[column.accessor]}>
                        {renderCell(column, row)}
                      </Tooltip>
                    ) : (
                      renderCell(column, row)
                    )}
                  </TableCell>
                ))}
                {actions && actions.length > 0 && (
                  <TableCell>
                    {actions.map((action, actionIndex) => (
                      action.condition?.(row) !== false && (
                        <Tooltip key={actionIndex} title={action.tooltip}>
                          <Icon
                            icon={action.icon}
                            size="sm"
                            onClick={() => action.onClick(row)}
                            className="cursor-pointer mx-1"
                          />
                        </Tooltip>
                      )
                    ))}
                  </TableCell>
                )}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length + (actions ? 1 : 0)}>
                <Text className="text-center">{emptyMessage}</Text>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      {deleteModal && deleteModal.isOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">
              &#8203;
            </span>

            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      {deleteModal.title}
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">{deleteModal.message}</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button onClick={deleteModal.onConfirm} color="red" className="ml-2">
                  Delete
                </Button>
                <Button onClick={deleteModal.onCancel}>Cancel</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default DataTable;
export type { Action, Column, DataTableProps, DeleteModalProps };
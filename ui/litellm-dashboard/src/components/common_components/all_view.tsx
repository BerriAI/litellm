import React from "react";
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
  onItemClick?: (item: any) => void;
}

const DataTable: React.FC<DataTableProps> = ({
  data,
  columns,
  actions,
  emptyMessage = "No data available",
  deleteModal,
  onItemClick,
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
              <Badge key={index} size="xs" className="mb-1" color="blue">
                <Text>{String(item).length > 30 ? `${String(item).slice(0, 30)}...` : item}</Text>
              </Badge>
            ))
          )}
        </div>
      );
    }

    return value?.toString() || "";
  };

  return (
    <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[40vh]">
      <Table>
        <TableHead>
          <TableRow>
            {columns.map((column, index) => (
              <TableHeaderCell key={index}>{column.header}</TableHeaderCell>
            ))}
            {actions && actions.length > 0 && <TableHeaderCell>Actions</TableHeaderCell>}
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
                      ...column.style,
                    }}
                  >
                    {column.accessor === "id" ? (
                      <Tooltip title={row[column.accessor]}>{renderCell(column, row)}</Tooltip>
                    ) : (
                      renderCell(column, row)
                    )}
                  </TableCell>
                ))}
                {actions && actions.length > 0 && (
                  <TableCell>
                    {actions.map(
                      (action, actionIndex) =>
                        // @ts-ignore
                        action.condition?.(row) !== false && (
                          <Tooltip key={actionIndex} title={action.tooltip}>
                            <Icon
                              // @ts-ignore
                              icon={action.icon}
                              size="sm"
                              onClick={() => action.onClick(row)}
                              className="cursor-pointer mx-1"
                            />
                          </Tooltip>
                        ),
                    )}
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
    </Card>
  );
};

export default DataTable;
export type { Action, Column, DataTableProps, DeleteModalProps };

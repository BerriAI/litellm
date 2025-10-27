import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Icon,
  Button,
  Badge,
  Text,
} from "@tremor/react";
import {
  PencilAltIcon,
  TrashIcon,
  SwitchVerticalIcon,
  ChevronUpIcon,
  ChevronDownIcon,
} from "@heroicons/react/outline";
import { Tooltip } from "antd";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Tag } from "./types";

interface TagTableProps {
  data: Tag[];
  onEdit: (tag: Tag) => void;
  onDelete: (tagName: string) => void;
  onSelectTag: (tagName: string) => void;
}

const TagTable: React.FC<TagTableProps> = ({
  data,
  onEdit,
  onDelete,
  onSelectTag,
}) => {
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: "created_at", desc: true }
  ]);

  const columns: ColumnDef<Tag>[] = [
    {
      header: "Tag Name",
      accessorKey: "name",
      cell: ({ row }) => {
        const tag = row.original;
        return (
          <div className="overflow-hidden">
            <Tooltip title={tag.name}>
              <Button
                size="xs"
                variant="light"
                className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5"
                onClick={() => onSelectTag(tag.name)}
              >
                {tag.name}
              </Button>
            </Tooltip>
          </div>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "description",
      cell: ({ row }) => {
        const tag = row.original;
        return (
          <Tooltip title={tag.description}>
            <span className="text-xs">
              {tag.description || "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      header: "Allowed LLMs",
      accessorKey: "models",
      cell: ({ row }) => {
        const tag = row.original;
        return (
          <div style={{ display: "flex", flexDirection: "column" }}>
            {tag?.models?.length === 0 ? (
              <Badge size="xs" className="mb-1" color="red">
                All Models
              </Badge>
            ) : (
              tag?.models?.map((modelId) => (
                <Badge
                  key={modelId}
                  size="xs"
                  className="mb-1"
                  color="blue"
                >
                  <Tooltip title={`ID: ${modelId}`}>
                    <Text>
                      {tag.model_info?.[modelId] || modelId}
                    </Text>
                  </Tooltip>
                </Badge>
              ))
            )}
          </div>
        );
      },
    },
    {
      header: "Created",
      accessorKey: "created_at",
      sortingFn: "datetime",
      cell: ({ row }) => {
        const tag = row.original;
        return (
          <span className="text-xs">
            {new Date(tag.created_at).toLocaleDateString()}
          </span>
        );
      },
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => {
        const tag = row.original;
        return (
          <div className="flex space-x-2">
            <Icon
              icon={PencilAltIcon}
              size="sm"
              onClick={() => onEdit(tag)}
              className="cursor-pointer"
            />
            <Icon
              icon={TrashIcon}
              size="sm"
              onClick={() => onDelete(tag.name)}
              className="cursor-pointer"
            />
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-0.5 [&_th]:py-1">
          <TableHead>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHeaderCell
                    key={header.id}
                    className={`py-1 h-8 ${
                      header.id === 'actions'
                        ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]'
                        : ''
                    }`}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder ? null : (
                          flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )
                        )}
                      </div>
                      {header.id !== 'actions' && (
                        <div className="w-4">
                          {header.column.getIsSorted() ? (
                            {
                              asc: <ChevronUpIcon className="h-4 w-4 text-blue-500" />,
                              desc: <ChevronDownIcon className="h-4 w-4 text-blue-500" />
                            }[header.column.getIsSorted() as string]
                          ) : (
                            <SwitchVerticalIcon className="h-4 w-4 text-gray-400" />
                          )}
                        </div>
                      )}
                    </div>
                  </TableHeaderCell>
                ))}
              </TableRow>
            ))}
          </TableHead>
          <TableBody>
            {table.getRowModel().rows.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${
                        cell.column.id === 'actions'
                          ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]'
                          : ''
                      }`}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
                    <p>No tags found</p>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default TagTable; 
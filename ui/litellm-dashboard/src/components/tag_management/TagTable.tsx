import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChevronDown,
  ChevronUp,
  ChevronsUpDown,
  Pencil,
  Trash2,
} from "lucide-react";
import React from "react";
import { Tag } from "./types";

interface TagTableProps {
  data: Tag[];
  onEdit: (tag: Tag) => void;
  onDelete: (tagName: string) => void;
  onSelectTag: (tagName: string) => void;
}

const DYNAMIC_SPEND_TAG_DESCRIPTION =
  "This is just a spend tag that was passed dynamically in a request. It does not control any LLM models.";

const TagTable: React.FC<TagTableProps> = ({
  data,
  onEdit,
  onDelete,
  onSelectTag,
}) => {
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: "created_at", desc: true },
  ]);

  const columns: ColumnDef<Tag>[] = [
    {
      header: "Tag Name",
      accessorKey: "name",
      cell: ({ row }) => {
        const tag = row.original;
        const isDynamicSpendTag = tag.description === DYNAMIC_SPEND_TAG_DESCRIPTION;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="font-mono text-primary bg-primary/10 hover:bg-primary/20 text-xs font-normal px-2 py-0.5 h-auto"
                  onClick={() => onSelectTag(tag.name)}
                  disabled={isDynamicSpendTag}
                >
                  {tag.name}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {isDynamicSpendTag
                  ? "You cannot view the information of a dynamically generated spend tag"
                  : tag.name}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "description",
      cell: ({ row }) => {
        const tag = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">{tag.description || "-"}</span>
              </TooltipTrigger>
              {tag.description && (
                <TooltipContent>{tag.description}</TooltipContent>
              )}
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Allowed Models",
      accessorKey: "models",
      cell: ({ row }) => {
        const tag = row.original;
        return (
          <div className="flex flex-col gap-1">
            {tag?.models?.length === 0 ? (
              <Badge variant="destructive">All Models</Badge>
            ) : (
              tag?.models?.map((modelId) => (
                <TooltipProvider key={modelId}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="secondary" className="w-fit">
                        {tag.model_info?.[modelId] || modelId}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>{`ID: ${modelId}`}</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
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
      cell: ({ row }) => (
        <span className="text-xs">
          {new Date(row.original.created_at).toLocaleDateString()}
        </span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        const tag = row.original;
        const isDynamicSpendTag = tag.description === DYNAMIC_SPEND_TAG_DESCRIPTION;
        return (
          <div className="flex space-x-1">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className={
                      isDynamicSpendTag
                        ? "opacity-50 cursor-not-allowed"
                        : "hover:text-primary"
                    }
                    disabled={isDynamicSpendTag}
                    onClick={() => onEdit(tag)}
                    aria-label="Edit tag"
                  >
                    <Pencil className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {isDynamicSpendTag
                    ? "Dynamically generated spend tags cannot be edited"
                    : "Edit tag"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className={
                      isDynamicSpendTag
                        ? "opacity-50 cursor-not-allowed"
                        : "hover:text-destructive"
                    }
                    disabled={isDynamicSpendTag}
                    onClick={() => onDelete(tag.name)}
                    aria-label="Delete tag"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {isDynamicSpendTag
                    ? "Dynamically generated spend tags cannot be deleted"
                    : "Delete tag"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-0.5 [&_th]:py-1">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    className={`py-1 h-8 ${
                      header.id === "actions"
                        ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                        : ""
                    }`}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext(),
                            )}
                      </div>
                      {header.id !== "actions" && (
                        <div className="w-4">
                          {header.column.getIsSorted() ? (
                            {
                              asc: (
                                <ChevronUp className="h-4 w-4 text-primary" />
                              ),
                              desc: (
                                <ChevronDown className="h-4 w-4 text-primary" />
                              ),
                            }[header.column.getIsSorted() as string]
                          ) : (
                            <ChevronsUpDown className="h-4 w-4 text-muted-foreground" />
                          )}
                        </div>
                      )}
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${
                        cell.column.id === "actions"
                          ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                      }`}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
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

import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Copy,
  Trash2,
} from "lucide-react";
import { PromptSpec, modelHubCall } from "@/components/networking";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { extractModel, getProviderFromModelHub } from "./prompt_utils";

interface PromptTableProps {
  promptsList: PromptSpec[];
  isLoading: boolean;
  onPromptClick?: (id: string) => void;
  onDeleteClick?: (id: string, name: string) => void;
  accessToken: string | null;
  isAdmin: boolean;
}

interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

const PromptTable: React.FC<PromptTableProps> = ({
  promptsList,
  isLoading,
  onPromptClick,
  onDeleteClick,
  accessToken,
  isAdmin,
}) => {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "created_at", desc: true },
  ]);
  const [modelHubData, setModelHubData] = useState<Map<string, ModelGroupInfo>>(
    new Map(),
  );

  useEffect(() => {
    const fetchModelHubData = async () => {
      if (!accessToken) return;

      try {
        const response = await modelHubCall(accessToken);
        if (response?.data) {
          const modelMap = new Map<string, ModelGroupInfo>();
          response.data.forEach((model: ModelGroupInfo) => {
            modelMap.set(model.model_group, model);
          });
          setModelHubData(modelMap);
        }
      } catch (error) {
        console.error("Error fetching model hub data:", error);
      }
    };

    fetchModelHubData();
  }, [accessToken]);

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const columns: ColumnDef<PromptSpec>[] = [
    {
      header: "Prompt ID",
      accessorKey: "prompt_id",
      cell: (info) => {
        const fullId = String(info.getValue() || "");
        const displayId =
          fullId.length > 25 ? `${fullId.slice(0, 25)}...` : fullId;
        return (
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate min-w-[220px] rounded"
                    onClick={() => fullId && onPromptClick?.(fullId)}
                  >
                    {displayId}
                  </button>
                </TooltipTrigger>
                <TooltipContent>{fullId}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      copyToClipboard(fullId);
                    }}
                    className="cursor-pointer text-muted-foreground hover:text-primary"
                    aria-label="Copy prompt ID"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Copy prompt ID</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        );
      },
    },
    {
      header: "Model",
      accessorKey: "model",
      cell: ({ row }) => {
        const prompt = row.original;
        const model = extractModel(prompt);

        if (!model) {
          return <span className="text-xs text-muted-foreground">-</span>;
        }

        const provider = getProviderFromModelHub(model, modelHubData);
        const { logo } = getProviderLogoAndName(provider || "");

        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center space-x-2">
                  <div className="flex-shrink-0">
                    {provider && logo ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={logo}
                        alt={`${provider} logo`}
                        className="w-4 h-4"
                        onError={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          const parent = target.parentElement;
                          if (!parent || !parent.contains(target)) {
                            return;
                          }

                          try {
                            const fallbackDiv = document.createElement("div");
                            fallbackDiv.className =
                              "w-4 h-4 rounded-full bg-muted flex items-center justify-center text-xs";
                            fallbackDiv.textContent =
                              provider?.charAt(0) || "-";
                            parent.replaceChild(fallbackDiv, target);
                          } catch (error) {
                            console.error(
                              "Failed to replace provider logo fallback:",
                              error,
                            );
                          }
                        }}
                      />
                    ) : (
                      <div className="w-4 h-4 rounded-full bg-muted flex items-center justify-center text-xs">
                        -
                      </div>
                    )}
                  </div>

                  <span className="max-w-[15ch] truncate block">{model}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>{model}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      cell: ({ row }) => {
        const prompt = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">{formatDate(prompt.created_at)}</span>
              </TooltipTrigger>
              <TooltipContent>{prompt.created_at}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      cell: ({ row }) => {
        const prompt = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">{formatDate(prompt.updated_at)}</span>
              </TooltipTrigger>
              <TooltipContent>{prompt.updated_at}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Environment",
      accessorKey: "environment",
      cell: ({ row }) => {
        const prompt = row.original;
        const env = prompt.environment || "development";
        const colorMap: Record<string, string> = {
          production:
            "text-red-600 bg-red-50 dark:text-red-300 dark:bg-red-950/30",
          staging:
            "text-amber-600 bg-amber-50 dark:text-amber-300 dark:bg-amber-950/30",
          development:
            "text-emerald-600 bg-emerald-50 dark:text-emerald-300 dark:bg-emerald-950/30",
        };
        return (
          <span
            className={cn(
              "text-xs px-2 py-0.5 rounded",
              colorMap[env] || "text-muted-foreground bg-muted",
            )}
          >
            {env}
          </span>
        );
      },
    },
    {
      header: "Created By",
      accessorKey: "created_by",
      cell: ({ row }) => {
        const prompt = row.original;
        return (
          <span className="text-xs text-muted-foreground">
            {prompt.created_by || "-"}
          </span>
        );
      },
    },
    {
      header: "Type",
      accessorKey: "prompt_info.prompt_type",
      cell: ({ row }) => {
        const prompt = row.original;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">
                  {prompt.prompt_info.prompt_type}
                </span>
              </TooltipTrigger>
              <TooltipContent>{prompt.prompt_info.prompt_type}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    ...(isAdmin
      ? [
          {
            header: "Actions",
            id: "actions",
            enableSorting: false,
            cell: ({ row }: { row: { original: PromptSpec } }) => {
              const prompt = row.original;
              const promptName = prompt.prompt_id || "Unknown Prompt";

              return (
                <div className="flex items-center gap-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteClick?.(prompt.prompt_id, promptName);
                          }}
                          aria-label="Delete prompt"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Delete prompt</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              );
            },
          },
        ]
      : []),
  ];

  const table = useReactTable({
    data: promptsList,
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
                    className="py-1 h-8"
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
                          <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
                    <p>Loading...</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : promptsList.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap"
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
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
                    <p>No prompts found</p>
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

export default PromptTable;

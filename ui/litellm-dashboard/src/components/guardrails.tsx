import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Text,
  Button,
  Icon,
} from "@tremor/react";
import {
  PlusIcon,
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
import { getGuardrailsList } from "./networking";
import AddGuardrailForm from "./guardrails/add_guardrail_form";
import { getGuardrailLogoAndName } from "./guardrails/guardrail_info_helpers";

interface GuardrailsPanelProps {
  accessToken: string | null;
}

interface GuardrailItem {
  guardrail_id?: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: string;
    default_on: boolean;
  };
  guardrail_info: Record<string, any> | null;
  created_at?: string;
  updated_at?: string;
}

interface GuardrailsResponse {
  guardrails: GuardrailItem[];
}

const GuardrailsPanel: React.FC<GuardrailsPanelProps> = ({ accessToken }) => {
  const [guardrailsList, setGuardrailsList] = useState<GuardrailItem[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([
    { id: "guardrail_name", desc: false }
  ]);

  const fetchGuardrails = async () => {
    if (!accessToken) {
      return;
    }
    
    setIsLoading(true);
    try {
      const response: GuardrailsResponse = await getGuardrailsList(accessToken);
      console.log(`guardrails: ${JSON.stringify(response)}`);
      setGuardrailsList(response.guardrails);
    } catch (error) {
      console.error('Error fetching guardrails:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGuardrails();
  }, [accessToken]);

  const handleAddGuardrail = () => {
    setIsAddModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleSuccess = () => {
    fetchGuardrails();
  };

  const handleDelete = (guardrailId: string) => {
    // Implement delete functionality here
    console.log(`Delete guardrail with ID: ${guardrailId}`);
  };

  const columns: ColumnDef<GuardrailItem>[] = [
    {
      header: "Name",
      accessorKey: "guardrail_name",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <Tooltip title={guardrail.guardrail_name}>
            <span className="text-xs font-medium">
              {guardrail.guardrail_name || "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      header: "Provider",
      accessorKey: "litellm_params.guardrail",
      cell: ({ row }) => {
        const guardrail = row.original;
        const { logo, displayName } = getGuardrailLogoAndName(guardrail.litellm_params.guardrail);
        return (
          <div className="flex items-center space-x-2">
            {logo && (
              <img 
                src={logo} 
                alt={`${displayName} logo`} 
                className="w-4 h-4"
                onError={(e) => {
                  // Hide broken image
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            )}
            <span className="text-xs">{displayName}</span>
          </div>
        );
      },
    },
    {
      header: "Mode",
      accessorKey: "litellm_params.mode",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <span className="text-xs">
            {guardrail.litellm_params.mode}
          </span>
        );
      },
    },
    {
      header: "Status",
      accessorKey: "litellm_params.default_on",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <div className={`inline-flex rounded-full px-2 py-1 text-xs font-medium
              ${guardrail.litellm_params.default_on 
              ? 'bg-green-100 text-green-800'  // Always On styling
              : 'bg-gray-100 text-gray-800'    // Per Request styling
              }`}>
              {guardrail.litellm_params.default_on ? 'Always On' : 'Per Request'}
          </div>
        );
      },
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => {
        const guardrail = row.original;
        return (
          <div className="flex space-x-2">
            <Icon
              icon={TrashIcon}
              size="sm"
              onClick={() => guardrail.guardrail_id && handleDelete(guardrail.guardrail_id)}
              className="cursor-pointer"
            />
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data: guardrailsList,
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
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex justify-between items-center mb-4">
        <Text className="text-lg">
          Configured guardrails and their current status. Setup guardrails in config.yaml or add them directly.{" "}
          <a 
              href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-blue-500 hover:text-blue-700 underline"
          >
              Docs
          </a>
        </Text>
        <Button 
          icon={PlusIcon} 
          onClick={handleAddGuardrail}
          disabled={!accessToken}
        >
          Add Guardrail
        </Button>
      </div>
      
      <Card>
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
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={columns.length} className="h-8 text-center">
                      <div className="text-center text-gray-500">
                        <p>Loading...</p>
                      </div>
                    </TableCell>
                  </TableRow>
                ) : table.getRowModel().rows.length > 0 ? (
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
                        <p>No guardrails found</p>
                      </div>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </div>
      </Card>

      <AddGuardrailForm 
        visible={isAddModalVisible}
        onClose={handleCloseModal}
        accessToken={accessToken}
        onSuccess={handleSuccess}
      />
    </div>
  );
};

export default GuardrailsPanel;
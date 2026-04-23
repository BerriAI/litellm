import React, { useState, useEffect } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  deletePassThroughEndpointsCall,
  getPassThroughEndpointsCall,
} from "./networking";
import { Eye, EyeOff, Info, Pencil, Trash2 } from "lucide-react";
import AddPassThroughEndpoint from "./add_pass_through";
import PassThroughInfoView from "./pass_through_info";
import { DataTable } from "./view_logs/table";
import { ColumnDef } from "@tanstack/react-table";
import NotificationsManager from "./molecules/notifications_manager";

interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelData: any;
  premiumUser?: boolean;
}

export interface passThroughItem {
  id?: string;
  path: string;
  target: string;
  headers: object;
  include_subpath?: boolean;
  cost_per_request?: number;
  auth?: boolean;
  methods?: string[];
  guardrails?: Record<
    string,
    { request_fields?: string[]; response_fields?: string[] } | null
  >;
  default_query_params?: Record<string, string>;
}

// Password field component for headers
const PasswordField: React.FC<{ value: object }> = ({ value }) => {
  const [showPassword, setShowPassword] = useState(false);
  const headerString = JSON.stringify(value);

  return (
    <div className="flex items-center space-x-2">
      <span className="font-mono text-xs">
        {showPassword ? headerString : "••••••••"}
      </span>
      <button
        onClick={() => setShowPassword(!showPassword)}
        className="p-1 hover:bg-muted rounded"
        type="button"
      >
        {showPassword ? (
          <EyeOff className="w-4 h-4 text-muted-foreground" />
        ) : (
          <Eye className="w-4 h-4 text-muted-foreground" />
        )}
      </button>
    </div>
  );
};

const PassThroughSettings: React.FC<GeneralSettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  modelData,
  premiumUser,
}) => {
  const [generalSettings, setGeneralSettings] = useState<passThroughItem[]>([]);
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(
    null,
  );
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [endpointToDelete, setEndpointToDelete] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getPassThroughEndpointsCall(accessToken).then((data) => {
      const general_settings = data["endpoints"];
      setGeneralSettings(general_settings);
    });
  }, [accessToken, userRole, userID]);

  const handleEndpointUpdated = () => {
    if (accessToken) {
      getPassThroughEndpointsCall(accessToken).then((data) => {
        const general_settings = data["endpoints"];
        setGeneralSettings(general_settings);
      });
    }
  };

  const handleDelete = async (endpointId: string) => {
    setEndpointToDelete(endpointId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (endpointToDelete == null || !accessToken) {
      return;
    }

    try {
      await deletePassThroughEndpointsCall(accessToken, endpointToDelete);

      const updatedSettings = generalSettings.filter(
        (setting) => setting.id !== endpointToDelete,
      );
      setGeneralSettings(updatedSettings);

      NotificationsManager.success("Endpoint deleted successfully.");
    } catch (error) {
      console.error("Error deleting the endpoint:", error);
      NotificationsManager.fromBackend(
        "Error deleting the endpoint: " + error,
      );
    }

    setIsDeleteModalOpen(false);
    setEndpointToDelete(null);
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setEndpointToDelete(null);
  };

  const handleResetField = (endpointId: string) => {
    handleDelete(endpointId);
  };

  const columns: ColumnDef<passThroughItem>[] = [
    {
      header: "ID",
      accessorKey: "id",
      cell: (info) => (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch]"
                onClick={() =>
                  info.row.original.id &&
                  setSelectedEndpointId(info.row.original.id)
                }
              >
                {info.row.original.id}
              </div>
            </TooltipTrigger>
            <TooltipContent>{info.row.original.id}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ),
    },
    {
      header: "Path",
      accessorKey: "path",
    },
    {
      header: "Target",
      accessorKey: "target",
      cell: (info) => (
        <span className="text-sm">{info.getValue() as string}</span>
      ),
    },
    {
      header: () => (
        <div className="flex items-center gap-1">
          <span>Methods</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                HTTP methods supported by this endpoint
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      ),
      accessorKey: "methods",
      cell: (info) => {
        const methods = info.getValue() as string[] | undefined;
        if (!methods || methods.length === 0) {
          return (
            <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
              ALL
            </Badge>
          );
        }
        return (
          <div className="flex flex-wrap gap-1">
            {methods.map((method: string) => (
              <Badge
                key={method}
                className="text-xs bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300"
              >
                {method}
              </Badge>
            ))}
          </div>
        );
      },
    },
    {
      header: () => (
        <div className="flex items-center gap-1">
          <span>Authentication</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                LiteLLM Virtual Key required to call endpoint
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      ),
      accessorKey: "auth",
      cell: (info) => (
        <Badge
          className={
            info.getValue()
              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
              : "bg-muted text-muted-foreground"
          }
        >
          {info.getValue() ? "Yes" : "No"}
        </Badge>
      ),
    },
    {
      header: "Headers",
      accessorKey: "headers",
      cell: (info) => <PasswordField value={(info.getValue() as object) || {}} />,
    },
    {
      header: "Actions",
      id: "actions",
      cell: ({ row }) => (
        <div className="flex space-x-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() =>
              row.original.id && setSelectedEndpointId(row.original.id)
            }
            title="Edit"
            aria-label="Edit"
          >
            <Pencil className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-destructive hover:text-destructive"
            onClick={() => handleResetField(row.original.id!)}
            title="Delete"
            aria-label="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      ),
    },
  ];

  if (!accessToken) {
    return null;
  }

  if (selectedEndpointId) {
    const selectedEndpoint = generalSettings.find(
      (endpoint) => endpoint.id === selectedEndpointId,
    );

    if (!selectedEndpoint) {
      return <div>Endpoint not found</div>;
    }

    return (
      <PassThroughInfoView
        endpointData={selectedEndpoint}
        onClose={() => setSelectedEndpointId(null)}
        accessToken={accessToken}
        isAdmin={userRole === "Admin" || userRole === "admin"}
        premiumUser={premiumUser}
        onEndpointUpdated={handleEndpointUpdated}
      />
    );
  }

  return (
    <div>
      <div>
        <h2 className="text-xl font-semibold">Pass Through Endpoints</h2>
        <p className="text-muted-foreground">
          Configure and manage your pass-through endpoints
        </p>
      </div>

      <AddPassThroughEndpoint
        accessToken={accessToken}
        setPassThroughItems={setGeneralSettings}
        passThroughItems={generalSettings}
        premiumUser={premiumUser}
      />

      <DataTable
        data={generalSettings}
        columns={columns}
        renderSubComponent={() => <div></div>}
        getRowCanExpand={() => false}
        isLoading={false}
        noDataMessage="No pass-through endpoints configured"
      />

      <AlertDialog
        open={isDeleteModalOpen}
        onOpenChange={(o) => (!o ? cancelDelete() : undefined)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Pass-Through Endpoint</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this pass-through endpoint? This
              action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={cancelDelete}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDelete}
              className="bg-destructive text-white hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default PassThroughSettings;

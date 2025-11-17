import React, { useState, useEffect } from "react";
import { Text, Button, Icon, Title } from "@tremor/react";
import { deletePassThroughEndpointsCall, getPassThroughEndpointsCall } from "./networking";
import { Badge, Tooltip } from "antd";
import { PencilAltIcon, TrashIcon, InformationCircleIcon } from "@heroicons/react/outline";
import AddPassThroughEndpoint from "./add_pass_through";
import PassThroughInfoView from "./pass_through_info";
import { DataTable } from "./view_logs/table";
import { ColumnDef } from "@tanstack/react-table";
import { Eye, EyeOff } from "lucide-react";
import NotificationsManager from "./molecules/notifications_manager";

interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
  premiumUser?: boolean;
}

interface routingStrategyArgs {
  ttl?: number;
  lowest_latency_buffer?: number;
}

interface nestedFieldItem {
  field_name: string;
  field_type: string;
  field_value: any;
  field_description: string;
  stored_in_db: boolean | null;
}

export interface passThroughItem {
  id?: string;
  path: string;
  target: string;
  headers: object;
  include_subpath?: boolean;
  cost_per_request?: number;
  auth?: boolean;
}

// Password field component for headers
const PasswordField: React.FC<{ value: object }> = ({ value }) => {
  const [showPassword, setShowPassword] = useState(false);
  const headerString = JSON.stringify(value);

  return (
    <div className="flex items-center space-x-2">
      <span className="font-mono text-xs">{showPassword ? headerString : "••••••••"}</span>
      <button onClick={() => setShowPassword(!showPassword)} className="p-1 hover:bg-gray-100 rounded" type="button">
        {showPassword ? <EyeOff className="w-4 h-4 text-gray-500" /> : <Eye className="w-4 h-4 text-gray-500" />}
      </button>
    </div>
  );
};

const PassThroughSettings: React.FC<GeneralSettingsPageProps> = ({ accessToken, userRole, userID, modelData, premiumUser }) => {
  const [generalSettings, setGeneralSettings] = useState<passThroughItem[]>([]);
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [endpointToDelete, setEndpointToDelete] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getPassThroughEndpointsCall(accessToken).then((data) => {
      let general_settings = data["endpoints"];
      setGeneralSettings(general_settings);
    });
  }, [accessToken, userRole, userID]);

  const handleEndpointUpdated = () => {
    // Refresh the endpoints list when an endpoint is updated
    if (accessToken) {
      getPassThroughEndpointsCall(accessToken).then((data) => {
        let general_settings = data["endpoints"];
        setGeneralSettings(general_settings);
      });
    }
  };

  const handleDelete = async (endpointId: string) => {
    // Set the endpoint to delete and open the confirmation modal
    setEndpointToDelete(endpointId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (endpointToDelete == null || !accessToken) {
      return;
    }

    try {
      await deletePassThroughEndpointsCall(accessToken, endpointToDelete);

      const updatedSettings = generalSettings.filter((setting) => setting.id !== endpointToDelete);
      setGeneralSettings(updatedSettings);

      NotificationsManager.success("Endpoint deleted successfully.");
    } catch (error) {
      console.error("Error deleting the endpoint:", error);
      NotificationsManager.fromBackend("Error deleting the endpoint: " + error);
    }

    // Close the confirmation modal and reset the endpointToDelete
    setIsDeleteModalOpen(false);
    setEndpointToDelete(null);
  };

  const cancelDelete = () => {
    // Close the confirmation modal and reset the endpointToDelete
    setIsDeleteModalOpen(false);
    setEndpointToDelete(null);
  };

  const handleResetField = (endpointId: string, idx: number) => {
    // Use handleDelete instead of direct deletion
    handleDelete(endpointId);
  };

  // Define columns for the DataTable
  const columns: ColumnDef<passThroughItem>[] = [
    {
      header: "ID",
      accessorKey: "id",
      cell: (info: any) => (
        <Tooltip title={info.row.original.id}>
          <div
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch]"
            onClick={() => info.row.original.id && setSelectedEndpointId(info.row.original.id)}
          >
            {info.row.original.id}
          </div>
        </Tooltip>
      ),
    },
    {
      header: "Path",
      accessorKey: "path",
    },
    {
      header: "Target",
      accessorKey: "target",
      cell: (info: any) => <Text>{info.getValue()}</Text>,
    },
    {
      header: () => (
        <div className="flex items-center gap-1">
          <span>Authentication</span>
          <Tooltip title="LiteLLM Virtual Key required to call endpoint">
            <InformationCircleIcon className="w-4 h-4 text-gray-400 cursor-help" />
          </Tooltip>
        </div>
      ),
      accessorKey: "auth",
      cell: (info: any) => <Badge color={info.getValue() ? "green" : "gray"}>{info.getValue() ? "Yes" : "No"}</Badge>,
    },
    {
      header: "Headers",
      accessorKey: "headers",
      cell: (info: any) => <PasswordField value={info.getValue() || {}} />,
    },
    {
      header: "Actions",
      id: "actions",
      cell: ({ row }) => (
        <div className="flex space-x-1">
          <Icon
            icon={PencilAltIcon}
            size="sm"
            onClick={() => row.original.id && setSelectedEndpointId(row.original.id)}
            title="Edit"
          />
          <Icon
            icon={TrashIcon}
            size="sm"
            onClick={() => handleResetField(row.original.id!, row.index)}
            title="Delete"
          />
        </div>
      ),
    },
  ];

  if (!accessToken) {
    return null;
  }

  // If a specific endpoint is selected, show the info view
  if (selectedEndpointId) {
    // Find the endpoint by ID to get the endpoint data for the info view
    console.log("selectedEndpointId", selectedEndpointId);
    console.log("generalSettings", generalSettings);
    const selectedEndpoint = generalSettings.find((endpoint) => endpoint.id === selectedEndpointId);

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
        <Title>Pass Through Endpoints</Title>
        <Text className="text-tremor-content">Configure and manage your pass-through endpoints</Text>
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

      {isDeleteModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            {/* Modal Panel */}
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">
              &#8203;
            </span>

            {/* Confirmation Modal Content */}
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Delete Pass-Through Endpoint</h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to delete this pass-through endpoint? This action cannot be undone.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button onClick={confirmDelete} color="red" className="ml-2">
                  Delete
                </Button>
                <Button onClick={cancelDelete}>Cancel</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PassThroughSettings;

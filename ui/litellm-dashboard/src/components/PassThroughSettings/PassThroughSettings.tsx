import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { deletePassThroughEndpointsCall, getPassThroughEndpointsCall } from "../networking";
import AddPassThroughEndpoint from "../add_pass_through";
import PassThroughInfoView from "../pass_through_info";
import NotificationsManager from "../molecules/notifications_manager";
import { PassThroughEndpointsTable } from "./PassThroughEndpointsTable";

interface PassThroughSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser?: boolean;
}

export interface passThroughItem {
  id?: string;
  path: string;
  target: string;
  headers: object;
  include_subpath?: boolean;
  cost_per_request?: number;
  timeout?: number;
  auth?: boolean;
  methods?: string[];
  guardrails?: Record<string, { request_fields?: string[]; response_fields?: string[] } | null>;
  default_query_params?: Record<string, string>;
}

const PassThroughSettings: React.FC<PassThroughSettingsProps> = ({ accessToken, userRole, userID, premiumUser }) => {
  const [generalSettings, setGeneralSettings] = useState<passThroughItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [endpointToDelete, setEndpointToDelete] = useState<string | null>(null);

  useEffect(() => {
    const fetchEndpoints = async () => {
      if (!accessToken || !userRole || !userID) {
        setIsLoading(false);
        return;
      }
      try {
        const data = await getPassThroughEndpointsCall(accessToken);
        setGeneralSettings(data["endpoints"]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchEndpoints();
  }, [accessToken, userRole, userID]);

  const handleEndpointUpdated = () => {
    if (accessToken) {
      getPassThroughEndpointsCall(accessToken).then((data) => {
        setGeneralSettings(data["endpoints"]);
      });
    }
  };

  const handleDelete = (endpointId: string) => {
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

    setIsDeleteModalOpen(false);
    setEndpointToDelete(null);
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setEndpointToDelete(null);
  };

  if (!accessToken) {
    return null;
  }

  if (selectedEndpointId) {
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
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-foreground">Pass Through Endpoints</h2>
        <p className="text-sm text-muted-foreground">Configure and manage your pass-through endpoints</p>
      </div>

      <AddPassThroughEndpoint
        accessToken={accessToken}
        setPassThroughItems={setGeneralSettings}
        passThroughItems={generalSettings}
        premiumUser={premiumUser}
      />

      <PassThroughEndpointsTable
        endpoints={generalSettings}
        isLoading={isLoading}
        onEndpointClick={setSelectedEndpointId}
        onDeleteClick={handleDelete}
      />

      {isDeleteModalOpen && (
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
                <Button variant="destructive" onClick={confirmDelete} className="ml-2">
                  Delete
                </Button>
                <Button variant="outline" onClick={cancelDelete}>
                  Cancel
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PassThroughSettings;

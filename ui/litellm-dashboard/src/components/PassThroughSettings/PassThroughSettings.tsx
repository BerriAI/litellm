import React, { useState, useEffect } from "react";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { deletePassThroughEndpointsCall, getPassThroughEndpointsCall } from "../networking";
import AddPassThroughEndpoint from "../add_pass_through";
import PassThroughInfoView from "../pass_through_info";
import { toast } from "@/lib/toast";
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
  is_from_config?: boolean;
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

      toast.success("Endpoint deleted successfully.");
    } catch (error) {
      console.error("Error deleting the endpoint:", error);
      toast.fromError("Error deleting the endpoint: " + error);
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

      <AlertDialog open={isDeleteModalOpen} onOpenChange={(open) => !open && cancelDelete()}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Pass-Through Endpoint</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this pass-through endpoint? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <Button variant="destructive" onClick={confirmDelete}>
              Delete
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default PassThroughSettings;

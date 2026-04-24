import React, { useCallback, useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDown, Code, Plus } from "lucide-react";
import { getGuardrailsList, deleteGuardrailCall } from "./networking";
import AddGuardrailForm from "./guardrails/add_guardrail_form";
import GuardrailTable from "./guardrails/guardrail_table";
import { isAdminRole } from "@/utils/roles";
import GuardrailInfoView from "./guardrails/guardrail_info";
import GuardrailTestPlayground from "./guardrails/GuardrailTestPlayground";
import NotificationsManager from "./molecules/notifications_manager";
import { Guardrail, GuardrailDefinitionLocation } from "./guardrails/types";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import { getGuardrailLogoAndName } from "./guardrails/guardrail_info_helpers";
import { CustomCodeModal } from "./guardrails/custom_code";
import GuardrailGarden from "./guardrails/guardrail_garden";
import { TeamGuardrailsTab } from "./guardrails/TeamGuardrailsTab";

interface GuardrailsPanelProps {
  accessToken: string | null;
  userRole?: string;
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
  guardrail_definition_location: GuardrailDefinitionLocation;
}

interface GuardrailsResponse {
  guardrails: Guardrail[];
}

const GuardrailsPanel: React.FC<GuardrailsPanelProps> = ({ accessToken, userRole }) => {
  const [guardrailsList, setGuardrailsList] = useState<Guardrail[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isCustomCodeModalVisible, setIsCustomCodeModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [guardrailToDelete, setGuardrailToDelete] = useState<Guardrail | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedGuardrailId, setSelectedGuardrailId] = useState<string | null>(null);
  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchGuardrails = useCallback(async () => {
    if (!accessToken) {
      return;
    }

    setIsLoading(true);
    try {
      const response: GuardrailsResponse = await getGuardrailsList(accessToken);
      setGuardrailsList(response.guardrails);
    } catch (error) {
      console.error("Error fetching guardrails:", error);
    } finally {
      setIsLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchGuardrails();
  }, [fetchGuardrails]);

  const handleAddGuardrail = () => {
    if (selectedGuardrailId) {
      setSelectedGuardrailId(null);
    }
    setIsAddModalVisible(true);
  };

  const handleAddCustomCodeGuardrail = () => {
    if (selectedGuardrailId) {
      setSelectedGuardrailId(null);
    }
    setIsCustomCodeModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleCloseCustomCodeModal = () => {
    setIsCustomCodeModalVisible(false);
  };

  const handleSuccess = () => {
    fetchGuardrails();
  };

  const handleDeleteClick = (guardrailId: string, _guardrailName: string) => {
    const guardrail =
      guardrailsList.find((g) => g.guardrail_id === guardrailId) || null;
    setGuardrailToDelete(guardrail);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!guardrailToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deleteGuardrailCall(accessToken, guardrailToDelete.guardrail_id);
      NotificationsManager.success(`Guardrail "${guardrailToDelete.guardrail_name}" deleted successfully`);
      await fetchGuardrails();
    } catch (error) {
      console.error("Error deleting guardrail:", error);
      NotificationsManager.fromBackend("Failed to delete guardrail");
    } finally {
      setIsDeleting(false);
      setIsDeleteModalOpen(false);
      setGuardrailToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalOpen(false);
    setGuardrailToDelete(null);
  };

  const providerDisplayName =
    guardrailToDelete && guardrailToDelete.litellm_params
      ? getGuardrailLogoAndName(guardrailToDelete.litellm_params.guardrail).displayName
      : undefined;

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <Tabs defaultValue="submitted">
        <TabsList>
          {isAdmin && (
            <>
              <TabsTrigger value="garden">Guardrail Garden</TabsTrigger>
              <TabsTrigger value="guardrails">Guardrails</TabsTrigger>
              <TabsTrigger value="playground" disabled={!accessToken}>
                Test Playground
              </TabsTrigger>
            </>
          )}
          <TabsTrigger value="submitted">Submitted Guardrails</TabsTrigger>
        </TabsList>

        {isAdmin && (
          <>
            <TabsContent value="garden">
              <GuardrailGarden
                accessToken={accessToken}
                onGuardrailCreated={handleSuccess}
              />
            </TabsContent>
            <TabsContent value="guardrails">
              <div className="flex justify-between items-center mb-4">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" disabled={!accessToken}>
                      + Add New Guardrail
                      <ChevronDown className="h-4 w-4 ml-1" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start">
                    <DropdownMenuItem onSelect={handleAddGuardrail}>
                      <Plus className="h-4 w-4" />
                      Add Provider Guardrail
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={handleAddCustomCodeGuardrail}>
                      <Code className="h-4 w-4" />
                      Create Custom Code Guardrail
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {selectedGuardrailId ? (
                <GuardrailInfoView
                  guardrailId={selectedGuardrailId}
                  onClose={() => setSelectedGuardrailId(null)}
                  accessToken={accessToken}
                  isAdmin={isAdmin}
                />
              ) : (
                <GuardrailTable
                  guardrailsList={guardrailsList}
                  isLoading={isLoading}
                  onDeleteClick={handleDeleteClick}
                  accessToken={accessToken}
                  onGuardrailUpdated={fetchGuardrails}
                  isAdmin={isAdmin}
                  onGuardrailClick={(id) => setSelectedGuardrailId(id)}
                />
              )}

              <AddGuardrailForm
                visible={isAddModalVisible}
                onClose={handleCloseModal}
                accessToken={accessToken}
                onSuccess={handleSuccess}
              />

              <CustomCodeModal
                visible={isCustomCodeModalVisible}
                onClose={handleCloseCustomCodeModal}
                accessToken={accessToken}
                onSuccess={handleSuccess}
              />

              <DeleteResourceModal
                isOpen={isDeleteModalOpen}
                title="Delete Guardrail"
                message={`Are you sure you want to delete guardrail: ${guardrailToDelete?.guardrail_name}? This action cannot be undone.`}
                resourceInformationTitle="Guardrail Information"
                resourceInformation={[
                  {
                    label: "Name",
                    value: guardrailToDelete?.guardrail_name,
                  },
                  {
                    label: "ID",
                    value: guardrailToDelete?.guardrail_id,
                    code: true,
                  },
                  { label: "Provider", value: providerDisplayName },
                  {
                    label: "Mode",
                    value: guardrailToDelete?.litellm_params.mode,
                  },
                  {
                    label: "Default On",
                    value: guardrailToDelete?.litellm_params.default_on
                      ? "Yes"
                      : "No",
                  },
                ]}
                onCancel={handleDeleteCancel}
                onOk={handleDeleteConfirm}
                confirmLoading={isDeleting}
              />
            </TabsContent>
            <TabsContent value="playground">
              <GuardrailTestPlayground
                guardrailsList={guardrailsList}
                isLoading={isLoading}
                accessToken={accessToken}
                onClose={() => {}}
              />
            </TabsContent>
          </>
        )}

        <TabsContent value="submitted">
          <TeamGuardrailsTab accessToken={accessToken} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default GuardrailsPanel;

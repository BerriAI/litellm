import React, { useState, useEffect } from "react";
import { Tabs } from "antd";
import { ChevronDown, Code, Plus } from "lucide-react";
import { getGuardrailsList, deleteGuardrailCall } from "@/components/networking";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import AddGuardrailForm from "./add_guardrail_form";
import GuardrailTable from "./guardrail_table";
import { isAdminRole } from "@/utils/roles";
import GuardrailInfoView from "./guardrail_info";
import GuardrailTestPlayground from "./GuardrailTestPlayground";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Guardrail } from "@/components/guardrails/types";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { getGuardrailLogoAndName } from "./guardrail_info_helpers";
import { CustomCodeModal } from "./custom_code";
import GuardrailGarden from "./guardrail_garden";
import { TeamGuardrailsTab } from "./TeamGuardrailsTab";

interface GuardrailsPanelProps {
  accessToken: string | null;
  userRole?: string;
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

  const fetchGuardrails = async () => {
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
  };

  useEffect(() => {
    fetchGuardrails();
  }, [accessToken]);

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

  const handleDeleteClick = (guardrailId: string, guardrailName: string) => {
    const guardrail = guardrailsList.find((g) => g.guardrail_id === guardrailId) || null;
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
      <Tabs
        defaultActiveKey="guardrails"
        items={[
          ...(isAdmin
            ? [
                {
                  key: "garden",
                  label: "Guardrail Garden",
                  children: <GuardrailGarden accessToken={accessToken} onGuardrailCreated={handleSuccess} />,
                },
                {
                  key: "guardrails",
                  label: "Guardrails",
                  children: (
                    <>
                      <div className="flex justify-between items-center mb-4">
                        <DropdownMenu>
                          <DropdownMenuTrigger
                            disabled={!accessToken}
                            className={cn(buttonVariants({ variant: "default" }))}
                          >
                            <Plus />
                            Add New Guardrail
                            <ChevronDown />
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="start" className="w-56">
                            <DropdownMenuItem onClick={handleAddGuardrail}>
                              <Plus />
                              Add Provider Guardrail
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={handleAddCustomCodeGuardrail}>
                              <Code />
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
                          onGuardrailUpdated={fetchGuardrails}
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
                          { label: "Name", value: guardrailToDelete?.guardrail_name },
                          { label: "ID", value: guardrailToDelete?.guardrail_id, code: true },
                          { label: "Provider", value: providerDisplayName },
                          { label: "Mode", value: guardrailToDelete?.litellm_params.mode },
                          {
                            label: "Default On",
                            value: guardrailToDelete?.litellm_params.default_on ? "Yes" : "No",
                          },
                        ]}
                        onCancel={handleDeleteCancel}
                        onOk={handleDeleteConfirm}
                        confirmLoading={isDeleting}
                      />
                    </>
                  ),
                },
                {
                  key: "playground",
                  label: "Test Playground",
                  disabled: !accessToken,
                  children: (
                    <GuardrailTestPlayground
                      guardrailsList={guardrailsList}
                      isLoading={isLoading}
                      accessToken={accessToken}
                      onClose={() => {}}
                    />
                  ),
                },
              ]
            : []),
          {
            key: "submitted",
            label: "Submitted Guardrails",
            children: <TeamGuardrailsTab accessToken={accessToken} />,
          },
        ]}
      />
    </div>
  );
};

export default GuardrailsPanel;

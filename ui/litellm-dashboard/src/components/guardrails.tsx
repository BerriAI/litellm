import React, { useState, useEffect } from "react";
import { Button, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { Modal } from "antd";
import { getGuardrailsList, deleteGuardrailCall } from "./networking";
import AddGuardrailForm from "./guardrails/add_guardrail_form";
import GuardrailTable from "./guardrails/guardrail_table";
import { isAdminRole } from "@/utils/roles";
import GuardrailInfoView from "./guardrails/guardrail_info";
import GuardrailTestPlayground from "./guardrails/GuardrailTestPlayground";
import NotificationsManager from "./molecules/notifications_manager";
import { Guardrail, GuardrailDefinitionLocation } from "./guardrails/types";

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
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [guardrailToDelete, setGuardrailToDelete] = useState<{ id: string; name: string } | null>(null);
  const [selectedGuardrailId, setSelectedGuardrailId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<number>(0);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

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

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleSuccess = () => {
    fetchGuardrails();
  };

  const handleDeleteClick = (guardrailId: string, guardrailName: string) => {
    setGuardrailToDelete({ id: guardrailId, name: guardrailName });
  };

  const handleDeleteConfirm = async () => {
    if (!guardrailToDelete || !accessToken) return;

    // Log removed to maintain clean production code
    setIsDeleting(true);
    try {
      await deleteGuardrailCall(accessToken, guardrailToDelete.id);
      NotificationsManager.success(`Guardrail "${guardrailToDelete.name}" deleted successfully`);
      fetchGuardrails(); // Refresh the list
    } catch (error) {
      console.error("Error deleting guardrail:", error);
      NotificationsManager.fromBackend("Failed to delete guardrail");
    } finally {
      setIsDeleting(false);
      setGuardrailToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setGuardrailToDelete(null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <TabGroup index={activeTab} onIndexChange={setActiveTab}>
        <TabList className="mb-4">
          <Tab>Guardrails</Tab>
          <Tab disabled={!accessToken || guardrailsList.length === 0}>Test Playground</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <div className="flex justify-between items-center mb-4">
              <Button onClick={handleAddGuardrail} disabled={!accessToken}>
                + Add New Guardrail
              </Button>
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

            {guardrailToDelete && (
              <Modal
                title="Delete Guardrail"
                open={guardrailToDelete !== null}
                onOk={handleDeleteConfirm}
                onCancel={handleDeleteCancel}
                confirmLoading={isDeleting}
                okText="Delete"
                okButtonProps={{ danger: true }}
              >
                <p>Are you sure you want to delete guardrail: {guardrailToDelete.name} ?</p>
                <p>This action cannot be undone.</p>
              </Modal>
            )}
          </TabPanel>

          <TabPanel>
            <GuardrailTestPlayground
              guardrailsList={guardrailsList}
              isLoading={isLoading}
              accessToken={accessToken}
              onClose={() => setActiveTab(0)}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default GuardrailsPanel;

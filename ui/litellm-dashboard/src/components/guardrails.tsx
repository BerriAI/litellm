import React, { useState, useEffect } from "react";
import { Button, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { Dropdown } from "antd";
import { DownOutlined, PlusOutlined, CodeOutlined } from "@ant-design/icons";
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

    // Log removed to maintain clean production code
    setIsDeleting(true);
    try {
      await deleteGuardrailCall(accessToken, guardrailToDelete.guardrail_id);
      NotificationsManager.success(`Guardrail "${guardrailToDelete.guardrail_name}" deleted successfully`);
      await fetchGuardrails(); // Refresh the list
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
      <TabGroup index={activeTab} onIndexChange={setActiveTab}>
        <TabList className="mb-4">
          <Tab>Guardrails</Tab>
          <Tab disabled={!accessToken || guardrailsList.length === 0}>Test Playground</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <div className="flex justify-between items-center mb-4">
              <Dropdown
                menu={{
                  items: [
                    {
                      key: "provider",
                      icon: <PlusOutlined />,
                      label: "Add Provider Guardrail",
                      onClick: handleAddGuardrail,
                    },
                    {
                      key: "custom_code",
                      icon: <CodeOutlined />,
                      label: "Create Custom Code Guardrail",
                      onClick: handleAddCustomCodeGuardrail,
                    },
                  ],
                }}
                trigger={["click"]}
                disabled={!accessToken}
              >
                <Button disabled={!accessToken}>
                  + Add New Guardrail <DownOutlined className="ml-2" />
                </Button>
              </Dropdown>
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

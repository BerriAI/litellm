import React, { useState, useEffect, useMemo, useCallback } from "react";
import { Button, TabGroup, TabList, Tab, TabPanels, TabPanel, Text } from "@tremor/react";
import { Dropdown, Select, Tooltip } from "antd";
import { DownOutlined, PlusOutlined, CodeOutlined, QuestionCircleOutlined } from "@ant-design/icons";
import { getGuardrailsList, deleteGuardrailCall, teamListCall } from "./networking";
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

const { Option } = Select;

interface GuardrailsPanelProps {
  accessToken: string | null;
  userRole?: string;
  userID?: string | null;
}

interface Team {
  team_id: string;
  team_alias: string;
  metadata?: {
    guardrails?: string[];
    [key: string]: any;
  };
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

const GuardrailsPanel: React.FC<GuardrailsPanelProps> = ({ accessToken, userRole, userID }) => {
  const [guardrailsList, setGuardrailsList] = useState<Guardrail[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isCustomCodeModalVisible, setIsCustomCodeModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [guardrailToDelete, setGuardrailToDelete] = useState<Guardrail | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedGuardrailId, setSelectedGuardrailId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<number>(0);
  
  // Team filtering state
  const [teams, setTeams] = useState<Team[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<string>("all");
  const [filteredGuardrails, setFilteredGuardrails] = useState<Guardrail[]>([]);

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

  const fetchTeams = async () => {
    if (!accessToken) {
      return;
    }

    try {
      const teamsResponse = await teamListCall(accessToken, null, isAdmin ? null : userID);
      setTeams(teamsResponse || []);
    } catch (error) {
      console.error("Error fetching teams:", error);
    }
  };

  useEffect(() => {
    fetchGuardrails();
    fetchTeams();
  }, [accessToken]);

  // Get teams that have guardrails configured
  const teamsWithGuardrails = useMemo(() => {
    return teams.filter(team => 
      team.metadata?.guardrails && 
      Array.isArray(team.metadata.guardrails) && 
      team.metadata.guardrails.length > 0
    );
  }, [teams]);

  // Filter guardrails based on selected team
  const filterGuardrails = useCallback((teamId: string) => {
    if (!guardrailsList) {
      setFilteredGuardrails([]);
      return;
    }

    if (teamId === "all") {
      // Show all guardrails
      setFilteredGuardrails(guardrailsList);
    } else if (teamId === "global") {
      // Show only guardrails with default_on = true (global guardrails)
      setFilteredGuardrails(guardrailsList.filter(g => g.litellm_params?.default_on === true));
    } else {
      // Show guardrails assigned to the selected team
      const team = teams.find(t => t.team_id === teamId);
      const teamGuardrailNames = team?.metadata?.guardrails || [];
      setFilteredGuardrails(
        guardrailsList.filter(g => 
          g.guardrail_name && teamGuardrailNames.includes(g.guardrail_name)
        )
      );
    }
  }, [guardrailsList, teams]);

  // Handle team filter change
  const handleTeamChange = (teamId: string) => {
    setSelectedTeam(teamId);
    filterGuardrails(teamId);
  };

  // Initial and effect-based filtering
  useEffect(() => {
    filterGuardrails(selectedTeam);
  }, [guardrailsList, selectedTeam, filterGuardrails]);

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

  // Count guardrails for display
  const globalGuardrailsCount = useMemo(() => 
    guardrailsList.filter(g => g.litellm_params?.default_on === true).length,
    [guardrailsList]
  );

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

            {/* Team Filter Section */}
            <div className="w-full mb-6">
              <div className="flex flex-col space-y-4">
                <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4 border-2 border-gray-200">
                  <div className="flex items-center gap-4">
                    <Text className="text-lg font-semibold text-gray-900">Current Team:</Text>
                    <Select 
                      value={selectedTeam} 
                      onChange={handleTeamChange} 
                      style={{ width: 350 }}
                    >
                      <Option value="all">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          <span className="font-medium">All Guardrails ({guardrailsList.length})</span>
                        </div>
                      </Option>
                      <Option value="global">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                          <span className="font-medium">Global Guardrails ({globalGuardrailsCount})</span>
                        </div>
                      </Option>
                      {teamsWithGuardrails.map((team) => (
                        <Option key={team.team_id} value={team.team_id}>
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="font-medium">
                              {team.team_alias || team.team_id} ({team.metadata?.guardrails?.length || 0})
                            </span>
                          </div>
                        </Option>
                      ))}
                    </Select>
                  </div>
                </div>
              </div>
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
                guardrailsList={filteredGuardrails}
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

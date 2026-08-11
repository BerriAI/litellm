import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import AvailableTeamsPanel from "@/components/team/AvailableTeamsPanel";
import TeamInfoView from "@/components/team/TeamInfo";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isProxyAdminRole } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Accordion, AccordionBody, AccordionHeader, TextInput } from "@tremor/react";
import { Button, Form, Input, Layout, Modal, Select, Switch, Tabs, theme, Tooltip, Typography } from "antd";
import { Plus, Users } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { PageHeader } from "@/components/shared/PageHeader";
import { Button as UIButton } from "@/components/ui/button";
import { teamsTableKeys } from "@/app/(dashboard)/hooks/teams/useTeams";
import { parseAsString, useQueryState } from "nuqs";
import { TeamsTable } from "./TeamsPage/TeamsTable";
import AccessGroupSelector from "./common_components/AccessGroupSelector";
import MetadataKeyValueFields, { metadataPairsToObject } from "./common_components/MetadataKeyValueFields";
import { useTeamMetadataSchema } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import PassThroughRoutesSelector from "./common_components/PassThroughRoutesSelector";
import AgentSelector from "./agent_management/AgentSelector";
import ModelAliasManager from "./common_components/ModelAliasManager";
import PremiumLoggingSettings from "./common_components/PremiumLoggingSettings";
import RouterSettingsAccordion, { RouterSettingsAccordionValue } from "./common_components/RouterSettingsAccordion";
import { fetchAvailableModelsForTeamOrKey } from "./key_team_helpers/fetch_available_models_team_key";
import type { Team } from "./key_team_helpers/key_list";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "./mcp_server_management/MCPToolPermissions";
import NotificationsManager from "./molecules/notifications_manager";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { Organization, getGuardrailsList, getPoliciesList, teamDeleteCall } from "./networking";
import NumericalInput from "./shared/numerical_input";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import SearchToolSelector from "./search_tools/SearchToolSelector";
import { useTranslation } from "react-i18next";

interface TeamProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  premiumUser?: boolean;
}

import DeleteResourceModal from "./common_components/DeleteResourceModal";
import { teamCreateCall } from "./networking";
import { normalizeTeamModelSelection } from "./team/teamModelAccess";
import { ModelSelect } from "./ModelSelect/ModelSelect";

const canCreateOrManageTeams = (
  userRole: string | null,
  userID: string | null,
  organizations: Organization[] | null,
): boolean => {
  // Admin role always has permission
  if (userRole === "Admin") {
    return true;
  }

  // Check if user is an org_admin in any organization
  if (organizations && userID) {
    return organizations.some((org) =>
      org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
    );
  }

  return false;
};

const getAdminOrganizations = (
  userRole: string | null,
  userID: string | null,
  organizations: Organization[] | null,
): Organization[] => {
  // Global Admin can see all organizations
  if (userRole === "Admin") {
    return organizations || [];
  }

  // Org Admin can only see organizations they're an admin for
  if (organizations && userID) {
    return organizations.filter((org) =>
      org.members?.some((member) => member.user_id === userID && member.user_role === "org_admin"),
    );
  }

  return [];
};

// @deprecated
const Teams: React.FC<TeamProps> = ({ accessToken, userID, userRole, premiumUser = false }) => {
  const { t } = useTranslation("gateway");
  const { data: organizationsData } = useOrganizations();
  const organizations = organizationsData ?? null;
  const { data: teamMetadataSchemaFields = [], isLoading: isTeamMetadataSchemaLoading } = useTeamMetadataSchema();
  const queryClient = useQueryClient();
  const refreshTeams = () => queryClient.invalidateQueries({ queryKey: teamsTableKeys.all });
  const [currentOrg] = useState<Organization | null>(null);
  const [currentOrgForCreateTeam, setCurrentOrgForCreateTeam] = useState<Organization | null>(null);

  const [form] = Form.useForm();

  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [selectedTeamId, setSelectedTeamId] = useQueryState("team", parseAsString.withOptions({ history: "push" }));
  const [editTeam, setEditTeam] = useState<boolean>(false);

  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [teamToDelete, setTeamToDelete] = useState<Team | null>(null);
  const [isTeamDeleting, setIsTeamDeleting] = useState(false);
  // Add this state near the other useState declarations
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [policiesList, setPoliciesList] = useState<string[]>([]);
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const [routerSettings, setRouterSettings] = useState<RouterSettingsAccordionValue | null>(null);
  const [routerSettingsKey, setRouterSettingsKey] = useState<number>(0);

  useEffect(() => {
    form.setFieldValue("models", []);
  }, [currentOrgForCreateTeam, userModels]);

  // Handle organization preselection when modal opens
  useEffect(() => {
    if (isTeamModalVisible) {
      const adminOrgs = getAdminOrganizations(userRole, userID, organizations);
      const isOrgAdmin = userRole !== "Admin";

      // Org admins must scope a team to an org, so with exactly one we preselect it.
      // Proxy admins can create org-less teams, so the field stays optional regardless of org count.
      if (isOrgAdmin && adminOrgs.length === 1) {
        const org = adminOrgs[0];
        form.setFieldValue("organization_id", org.organization_id);
        setCurrentOrgForCreateTeam(org);
      } else {
        form.setFieldValue("organization_id", currentOrg?.organization_id || null);
        setCurrentOrgForCreateTeam(currentOrg);
      }
    }
  }, [isTeamModalVisible, userRole, userID, organizations, currentOrg]);

  // Add this useEffect to fetch guardrails
  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchPolicies = async () => {
      try {
        if (accessToken == null) {
          return;
        }

        const response = await getPoliciesList(accessToken);
        const policyNames = response.policies.map((p: { policy_name: string }) => p.policy_name);
        setPoliciesList(policyNames);
      } catch (error) {
        console.error("Failed to fetch policies:", error);
      }
    };

    fetchGuardrails();
    fetchPolicies();
  }, [accessToken]);

  const handleOk = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setModelAliases({});
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
  };

  const handleCancel = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setModelAliases({});
    setRouterSettings(null);
    setRouterSettingsKey((prev) => prev + 1);
  };

  const handleDelete = async (team: Team) => {
    // Set the team to delete and open the confirmation modal
    setTeamToDelete(team);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (teamToDelete == null || accessToken == null) {
      return;
    }

    try {
      setIsTeamDeleting(true);
      await teamDeleteCall(accessToken, teamToDelete.team_id);
      await refreshTeams();
      NotificationsManager.success(t("teams.notifications.deleted"));
    } catch (error) {
      NotificationsManager.fromBackend(t("teams.notifications.deleteFailed", { error: String(error) }));
    } finally {
      setIsTeamDeleting(false);
      setIsDeleteModalOpen(false);
      setTeamToDelete(null);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setTeamToDelete(null);
  };

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null || accessToken === null) {
          return;
        }
        const models = await fetchAvailableModelsForTeamOrKey(userID, userRole, accessToken);
        if (models) {
          setUserModels(models);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole]);

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null) {
        let organizationId = formValues?.organization_id || currentOrg?.organization_id;
        if (organizationId === "" || typeof organizationId !== "string") {
          formValues.organization_id = null;
        } else {
          formValues.organization_id = organizationId.trim();
        }

        NotificationsManager.info(t("teams.notifications.creating"));

        const metadataObject = {
          ...metadataPairsToObject(formValues.metadata),
          ...(loggingSettings.length > 0 ? { logging: loggingSettings.filter((config) => config.callback_name) } : {}),
        };
        formValues.metadata = Object.keys(metadataObject).length > 0 ? JSON.stringify(metadataObject) : undefined;

        if (formValues.secret_manager_settings) {
          if (typeof formValues.secret_manager_settings === "string") {
            if (formValues.secret_manager_settings.trim() === "") {
              delete formValues.secret_manager_settings;
            } else {
              try {
                formValues.secret_manager_settings = JSON.parse(formValues.secret_manager_settings);
              } catch (e) {
                throw new Error(t("teams.notifications.secretManagerParseFailed", { error: String(e) }));
              }
            }
          }
        }

        const hasSearchTools =
          Array.isArray(formValues.object_permission_search_tools) &&
          formValues.object_permission_search_tools.length > 0;

        if (
          (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) ||
          (formValues.allowed_mcp_servers_and_groups &&
            (formValues.allowed_mcp_servers_and_groups.servers?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.accessGroups?.length > 0 ||
              formValues.allowed_mcp_servers_and_groups.toolPermissions))
        ) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          if (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) {
            formValues.object_permission.vector_stores = formValues.allowed_vector_store_ids;
            delete formValues.allowed_vector_store_ids;
          }
          if (formValues.allowed_mcp_servers_and_groups) {
            const { servers, accessGroups } = formValues.allowed_mcp_servers_and_groups;
            if (servers && servers.length > 0) {
              formValues.object_permission.mcp_servers = servers;
            }
            if (accessGroups && accessGroups.length > 0) {
              formValues.object_permission.mcp_access_groups = accessGroups;
            }
            delete formValues.allowed_mcp_servers_and_groups;
          }

          if (formValues.mcp_tool_permissions && Object.keys(formValues.mcp_tool_permissions).length > 0) {
            formValues.object_permission.mcp_tool_permissions = formValues.mcp_tool_permissions;
            delete formValues.mcp_tool_permissions;
          }
        }

        // Transform allowed_mcp_access_groups into object_permission
        if (formValues.allowed_mcp_access_groups && formValues.allowed_mcp_access_groups.length > 0) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          formValues.object_permission.mcp_access_groups = formValues.allowed_mcp_access_groups;
          delete formValues.allowed_mcp_access_groups;
        }

        // Handle agent permissions
        if (formValues.allowed_agents_and_groups) {
          const { agents, accessGroups } = formValues.allowed_agents_and_groups;
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          if (agents && agents.length > 0) {
            formValues.object_permission.agents = agents;
          }
          if (accessGroups && accessGroups.length > 0) {
            formValues.object_permission.agent_access_groups = accessGroups;
          }
          delete formValues.allowed_agents_and_groups;
        }

        if (hasSearchTools) {
          if (!formValues.object_permission) {
            formValues.object_permission = {};
          }
          formValues.object_permission.search_tools = formValues.object_permission_search_tools;
          delete formValues.object_permission_search_tools;
        }

        // Add model_aliases if any are defined
        if (Object.keys(modelAliases).length > 0) {
          formValues.model_aliases = modelAliases;
        }

        // Add router_settings if any are defined
        if (routerSettings?.router_settings) {
          // Only include router_settings if it has at least one non-null value
          const hasValues = Object.values(routerSettings.router_settings).some(
            (value) => value !== null && value !== undefined && value !== "",
          );
          if (hasValues) {
            formValues.router_settings = routerSettings.router_settings;
          }
        }

        await teamCreateCall(accessToken, { ...formValues, models: normalizeTeamModelSelection(formValues.models) });
        NotificationsManager.success(t("teams.notifications.created"));
        await refreshTeams();
        form.resetFields();
        setLoggingSettings([]);
        setModelAliases({});
        setRouterSettings(null);
        setRouterSettingsKey((prev) => prev + 1);
        setIsTeamModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the team:", error);
      NotificationsManager.fromBackend(
        t("teams.notifications.createFailed", { error: extractProxyErrorMessage(error) }),
      );
    }
  };

  const is_team_admin = (team: any) => {
    if (team == null || team.members_with_roles == null) {
      return false;
    }
    for (let i = 0; i < team.members_with_roles.length; i++) {
      let member = team.members_with_roles[i];
      if (member.user_id == userID && member.role == "admin") {
        return true;
      }
    }
    return false;
  };

  const { token } = theme.useToken();
  const { Text } = Typography;
  const { Content } = Layout;

  const tabItems = [
    {
      key: "your-teams",
      label: t("teams.tabs.yours"),
      children: (
        <>
          <TeamsTable
            userRole={userRole}
            userID={userID}
            onSelectTeam={(team) => {
              setSelectedTeam(team);
              void setSelectedTeamId(team.team_id);
              setEditTeam(false);
            }}
            onEditTeam={(team) => {
              setSelectedTeam(team);
              void setSelectedTeamId(team.team_id);
              setEditTeam(true);
            }}
            onDeleteTeam={handleDelete}
          />

          <DeleteResourceModal
            isOpen={isDeleteModalOpen}
            title={t("teams.delete.title")}
            alertMessage={(() => {
              const deleteKeyCount = teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0;
              return deleteKeyCount === 0 ? undefined : t("teams.delete.warning", { count: deleteKeyCount });
            })()}
            message={t("teams.delete.message")}
            resourceInformationTitle={t("teams.delete.information")}
            resourceInformation={[
              { label: t("teams.table.teamId"), value: teamToDelete?.team_id, code: true },
              { label: t("teams.create.teamName"), value: teamToDelete?.team_alias },
              {
                label: t("teams.table.keys"),
                value: teamToDelete?.keys_count ?? teamToDelete?.keys?.length ?? 0,
              },
              { label: t("teams.table.members"), value: teamToDelete?.members_with_roles?.length },
            ]}
            requiredConfirmation={teamToDelete?.team_alias}
            onCancel={cancelDelete}
            onOk={confirmDelete}
            confirmLoading={isTeamDeleting}
          />
        </>
      ),
    },
    {
      key: "available-teams",
      label: t("teams.tabs.available"),
      children: <AvailableTeamsPanel accessToken={accessToken} userID={userID} />,
    },
    ...(isProxyAdminRole(userRole || "")
      ? [
          {
            key: "default-settings",
            label: t("teams.tabs.defaultSettings"),
            children: <TeamSSOSettings accessToken={accessToken} userID={userID || ""} userRole={userRole || ""} />,
          },
        ]
      : []),
  ];

  return (
    <Content style={{ padding: token.paddingLG, paddingInline: token.paddingLG * 2 }}>
      {selectedTeamId ? (
        <TeamInfoView
          teamId={selectedTeamId}
          onUpdate={() => {
            refreshTeams();
          }}
          onClose={() => {
            setSelectedTeam(null);
            void setSelectedTeamId(null);
            setEditTeam(false);
          }}
          accessToken={accessToken}
          is_team_admin={is_team_admin(selectedTeam?.team_id === selectedTeamId ? selectedTeam : null)}
          is_proxy_admin={userRole == "Admin"}
          userModels={userModels}
          editTeam={editTeam}
          premiumUser={premiumUser}
        />
      ) : (
        <>
          <div className="mb-4">
            <PageHeader icon={<Users className="size-5" />} title={t("teams.title")} subtitle={t("teams.subtitle")} />
          </div>

          <Tabs
            items={tabItems}
            tabBarExtraContent={{
              left: canCreateOrManageTeams(userRole, userID, organizations) ? (
                <div className="flex items-center gap-4 pr-4">
                  <UIButton onClick={() => setIsTeamModalVisible(true)} data-testid="create-team-button">
                    <Plus className="size-4" />
                    {t("teams.create.button")}
                  </UIButton>
                  <div className="h-6 w-px bg-gray-200" />
                </div>
              ) : undefined,
            }}
          />
        </>
      )}

      {canCreateOrManageTeams(userRole, userID, organizations) && (
        <Modal
          title={t("teams.create.title")}
          open={isTeamModalVisible}
          width={1000}
          footer={null}
          onOk={handleOk}
          onCancel={handleCancel}
          destroyOnHidden
        >
          <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
            <>
              <Form.Item
                label={t("teams.create.teamName")}
                name="team_alias"
                rules={[
                  {
                    required: true,
                    message: t("teams.create.teamNameRequired"),
                  },
                ]}
              >
                <TextInput placeholder="" data-testid="team-name-input" />
              </Form.Item>
              {(() => {
                const adminOrgs = getAdminOrganizations(userRole, userID, organizations);
                const isOrgAdmin = userRole !== "Admin";
                const isSingleOrg = adminOrgs.length === 1;
                const hasNoOrgs = adminOrgs.length === 0;

                return (
                  <>
                    <Form.Item
                      label={
                        <span>
                          {t("teams.create.organization")}{" "}
                          <Tooltip
                            title={
                              <span>
                                {t("teams.create.organizationTooltip")}{" "}
                                <a
                                  href="https://docs.litellm.ai/docs/proxy/user_management_heirarchy"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  style={{
                                    color: "#1890ff",
                                    textDecoration: "underline",
                                  }}
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  {t("teams.create.hierarchy")}
                                </a>
                              </span>
                            }
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      name="organization_id"
                      initialValue={currentOrg ? currentOrg.organization_id : null}
                      className="mt-8"
                      rules={
                        isOrgAdmin
                          ? [
                              {
                                required: true,
                                message: t("teams.create.organizationRequired"),
                              },
                            ]
                          : []
                      }
                      help={
                        isOrgAdmin && isSingleOrg
                          ? t("teams.create.singleOrganization")
                          : isOrgAdmin
                            ? t("teams.create.required")
                            : ""
                      }
                    >
                      <Select
                        showSearch
                        allowClear={!isOrgAdmin}
                        disabled={isOrgAdmin && isSingleOrg}
                        placeholder={
                          hasNoOrgs ? t("teams.create.noOrganizations") : t("teams.create.selectOrganization")
                        }
                        onChange={(value) => {
                          form.setFieldValue("organization_id", value);
                          setCurrentOrgForCreateTeam(adminOrgs?.find((org) => org.organization_id === value) || null);
                        }}
                        filterOption={(input, option) => {
                          if (!option) return false;
                          const optionValue = option.children?.toString() || "";
                          return optionValue.toLowerCase().includes(input.toLowerCase());
                        }}
                        optionFilterProp="children"
                      >
                        {adminOrgs?.map((org) => (
                          <Select.Option key={org.organization_id} value={org.organization_id}>
                            <span className="font-medium">{org.organization_alias}</span>{" "}
                            <span className="text-gray-500">({org.organization_id})</span>
                          </Select.Option>
                        ))}
                      </Select>
                    </Form.Item>

                    {/* Show message when org admin needs to select organization */}
                    {isOrgAdmin && !isSingleOrg && adminOrgs.length > 1 && (
                      <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-md">
                        <Text style={{ color: "#1e40af", fontSize: 14 }}>
                          {t("teams.create.organizationAdminHint")}
                        </Text>
                      </div>
                    )}
                  </>
                );
              })()}
              <Form.Item
                label={
                  <span>
                    {t("teams.create.models")}{" "}
                    <Tooltip title={t("teams.create.modelsTooltip")}>
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="models"
              >
                <ModelSelect
                  value={form.getFieldValue("models") || []}
                  onChange={(values) => form.setFieldValue("models", values)}
                  organizationID={form.getFieldValue("organization_id")}
                  options={{
                    includeSpecialOptions: true,
                    showAllProxyModelsOverride: !form.getFieldValue("organization_id"),
                  }}
                  context="team"
                  dataTestId="create-team-models-select"
                />
              </Form.Item>

              <Form.Item label={t("teams.create.maxBudget")} name="max_budget">
                <NumericalInput step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item className="mt-8" label={t("teams.create.resetBudget")} name="budget_duration">
                <Select defaultValue={null} placeholder={t("teams.create.notApplicable")}>
                  <Select.Option value="24h">{t("teams.create.daily")}</Select.Option>
                  <Select.Option value="7d">{t("teams.create.weekly")}</Select.Option>
                  <Select.Option value="30d">{t("teams.create.monthly")}</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item label={t("teams.create.tpmLimit")} name="tpm_limit">
                <NumericalInput step={1} width={400} />
              </Form.Item>
              <Form.Item label={t("teams.create.rpmLimit")} name="rpm_limit">
                <NumericalInput step={1} width={400} />
              </Form.Item>
              <Form.Item label={t("teams.create.metadata")} help={t("teams.create.metadataHelp")}>
                <MetadataKeyValueFields
                  form={form}
                  schemaFields={teamMetadataSchemaFields}
                  schemaLoading={isTeamMetadataSchemaLoading}
                />
              </Form.Item>

              <Accordion className="mt-20 mb-8">
                <AccordionHeader>
                  <b>{t("teams.create.additionalSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item label={t("teams.table.teamId")} name="team_id" help={t("teams.create.teamIdHelp")}>
                    <TextInput
                      onChange={(e) => {
                        e.target.value = e.target.value.trim();
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    label={t("teams.create.memberBudget")}
                    name="team_member_budget"
                    normalize={(value) => (value ? Number(value) : undefined)}
                    tooltip={t("teams.create.memberBudgetTooltip")}
                  >
                    <NumericalInput step={0.01} precision={2} width={200} />
                  </Form.Item>
                  <Form.Item
                    label={t("teams.create.memberKeyDuration")}
                    name="team_member_key_duration"
                    tooltip={t("teams.create.memberKeyDurationTooltip")}
                  >
                    <TextInput placeholder={t("teams.create.memberKeyDurationPlaceholder")} />
                  </Form.Item>
                  <Form.Item
                    label={t("teams.create.memberRpmLimit")}
                    name="team_member_rpm_limit"
                    tooltip={t("teams.create.memberRpmTooltip")}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <Form.Item
                    label={t("teams.create.memberTpmLimit")}
                    name="team_member_tpm_limit"
                    tooltip={t("teams.create.memberTpmTooltip")}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <Form.Item
                    label={t("teams.create.secretManager")}
                    name="secret_manager_settings"
                    help={premiumUser ? t("teams.create.secretManagerHelp") : t("teams.create.secretManagerPremium")}
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (!value) {
                            return Promise.resolve();
                          }
                          try {
                            JSON.parse(value);
                            return Promise.resolve();
                          } catch (error) {
                            return Promise.reject(new Error(t("teams.create.validJson")));
                          }
                        },
                      },
                    ]}
                  >
                    <Input.TextArea
                      rows={4}
                      placeholder='{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}'
                      disabled={!premiumUser}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.guardrails")}{" "}
                        <Tooltip title={t("teams.create.guardrailsSetup")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="guardrails"
                    className="mt-8"
                    help={t("teams.create.guardrailsHelp")}
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      placeholder={t("teams.create.guardrailsPlaceholder")}
                      options={guardrailsList.map((name) => ({
                        value: name,
                        label: name,
                      }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.disableGlobalGuardrails")}{" "}
                        <Tooltip title={t("teams.create.disableGlobalGuardrailsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="disable_global_guardrails"
                    className="mt-4"
                    valuePropName="checked"
                    help={t("teams.create.disableGlobalGuardrailsHelp")}
                  >
                    <Switch
                      disabled={!premiumUser}
                      checkedChildren={premiumUser ? t("teams.create.yes") : t("teams.create.disableGuardrailsPremium")}
                      unCheckedChildren={
                        premiumUser ? t("teams.create.no") : t("teams.create.disableGuardrailsPremium")
                      }
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.policies")}{" "}
                        <Tooltip title={t("teams.create.policiesTooltip")}>
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="policies"
                    className="mt-8"
                    help={t("teams.create.policiesHelp")}
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      placeholder={t("teams.create.policiesPlaceholder")}
                      options={policiesList.map((name) => ({
                        value: name,
                        label: name,
                      }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.accessGroups")}{" "}
                        <Tooltip title={t("teams.create.accessGroupsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="access_group_ids"
                    className="mt-8"
                    help={t("teams.create.accessGroupsHelp")}
                  >
                    <AccessGroupSelector placeholder={t("teams.create.accessGroupsPlaceholder")} />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.vectorStores")}{" "}
                        <Tooltip title={t("teams.create.vectorStoresTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_vector_store_ids"
                    className="mt-8"
                    help={t("teams.create.vectorStoresHelp")}
                  >
                    <VectorStoreSelector
                      onChange={(values: string[]) => form.setFieldValue("allowed_vector_store_ids", values)}
                      value={form.getFieldValue("allowed_vector_store_ids")}
                      accessToken={accessToken || ""}
                      placeholder={t("teams.create.vectorStoresPlaceholder")}
                    />
                  </Form.Item>
                  <Form.Item
                    label={t("teams.create.passthroughRoutes")}
                    name="allowed_passthrough_routes"
                    className="mt-8"
                    tooltip={
                      !premiumUser
                        ? t("teams.create.passthroughPremium")
                        : !isProxyAdminRole(userRole || "")
                          ? t("teams.create.passthroughAdminOnly")
                          : undefined
                    }
                  >
                    <PassThroughRoutesSelector
                      accessToken={accessToken || ""}
                      placeholder={t("teams.create.passthroughPlaceholder")}
                      disabled={!premiumUser || !isProxyAdminRole(userRole || "")}
                    />
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("teams.create.mcpSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.allowedMcp")}{" "}
                        <Tooltip title={t("teams.create.allowedMcpTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_mcp_servers_and_groups"
                    className="mt-4"
                    help={t("teams.create.allowedMcpHelp")}
                  >
                    <MCPServerSelector
                      onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
                      value={form.getFieldValue("allowed_mcp_servers_and_groups")}
                      accessToken={accessToken || ""}
                      placeholder={t("teams.create.allowedMcpPlaceholder")}
                      allowAllProxyMcpServers={isProxyAdminRole(userRole || "")}
                    />
                  </Form.Item>

                  {/* Hidden field to register mcp_tool_permissions with the form */}
                  <Form.Item name="mcp_tool_permissions" initialValue={{}} hidden>
                    <Input type="hidden" />
                  </Form.Item>

                  <Form.Item
                    noStyle
                    shouldUpdate={(prevValues, currentValues) =>
                      prevValues.allowed_mcp_servers_and_groups !== currentValues.allowed_mcp_servers_and_groups ||
                      prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
                    }
                  >
                    {() => (
                      <div className="mt-6">
                        <MCPToolPermissions
                          accessToken={accessToken || ""}
                          selectedServers={form.getFieldValue("allowed_mcp_servers_and_groups")?.servers || []}
                          toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
                          onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
                        />
                      </div>
                    )}
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("teams.create.agentSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.allowedAgents")}{" "}
                        <Tooltip title={t("teams.create.allowedAgentsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_agents_and_groups"
                    className="mt-4"
                    help={t("teams.create.allowedAgentsHelp")}
                  >
                    <AgentSelector
                      onChange={(val: any) => form.setFieldValue("allowed_agents_and_groups", val)}
                      value={form.getFieldValue("allowed_agents_and_groups")}
                      accessToken={accessToken || ""}
                      placeholder={t("teams.create.allowedAgentsPlaceholder")}
                    />
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("teams.create.searchToolSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    label={
                      <span>
                        {t("teams.create.allowedSearchTools")}{" "}
                        <Tooltip title={t("teams.create.allowedSearchToolsTooltip")}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="object_permission_search_tools"
                    className="mt-4"
                    help={t("teams.create.allowedSearchToolsHelp")}
                  >
                    <SearchToolSelector
                      onChange={(vals: string[]) => form.setFieldValue("object_permission_search_tools", vals)}
                      value={form.getFieldValue("object_permission_search_tools")}
                      accessToken={accessToken || ""}
                      placeholder={t("teams.create.allowedSearchToolsPlaceholder")}
                    />
                  </Form.Item>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("teams.create.loggingSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <div className="mt-4">
                    <PremiumLoggingSettings
                      value={loggingSettings}
                      onChange={setLoggingSettings}
                      premiumUser={premiumUser}
                    />
                  </div>
                </AccordionBody>
              </Accordion>

              <Accordion key={`router-settings-accordion-${routerSettingsKey}`} className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("teams.create.routerSettings")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <div className="mt-4 w-full">
                    <RouterSettingsAccordion
                      key={routerSettingsKey}
                      accessToken={accessToken || ""}
                      value={routerSettings || undefined}
                      onChange={setRouterSettings}
                      modelData={
                        userModels.length > 0 ? { data: userModels.map((model) => ({ model_name: model })) } : undefined
                      }
                    />
                  </div>
                </AccordionBody>
              </Accordion>

              <Accordion className="mt-8 mb-8">
                <AccordionHeader>
                  <b>{t("teams.create.modelAliases")}</b>
                </AccordionHeader>
                <AccordionBody>
                  <div className="mt-4">
                    <Text type="secondary" style={{ fontSize: 14, marginBottom: 16, display: "block" }}>
                      {t("teams.create.modelAliasesDescription")}
                    </Text>
                    <ModelAliasManager
                      accessToken={accessToken || ""}
                      initialModelAliases={modelAliases}
                      onAliasUpdate={setModelAliases}
                      showExampleConfig={false}
                    />
                  </div>
                </AccordionBody>
              </Accordion>
            </>
            <div style={{ textAlign: "right", marginTop: "10px" }}>
              <Button htmlType="submit" data-testid="create-team-submit">
                {t("teams.create.submit")}
              </Button>
            </div>
          </Form>
        </Modal>
      )}
    </Content>
  );
};

export default Teams;

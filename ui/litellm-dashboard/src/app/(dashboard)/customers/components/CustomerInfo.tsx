import React, { useEffect, useState } from "react";
import { Button, Form, Input, Tabs } from "antd";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { Badge, Card, Grid, Text, Title } from "@tremor/react";
import { CheckIcon, CopyIcon } from "lucide-react";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { customerUpdateCall } from "@/components/networking";
import type { Customer } from "@/app/(dashboard)/customers/types";
import CustomerFormFields from "@/app/(dashboard)/customers/components/CustomerFormFields";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import AgentSelector from "@/components/agent_management/AgentSelector";
import MCPToolPermissions from "@/components/mcp_server_management/MCPToolPermissions";
import ObjectPermissionsView from "@/components/object_permissions_view";

export interface CustomerInfoProps {
  customerId: string;
  initialCustomer: Customer | null;
  onClose: () => void;
  onUpdate: (customer: Customer) => void;
  accessToken: string | null;
  userRole: string | null;
  defaultTab?: "overview" | "settings";
}

const CustomerInfo: React.FC<CustomerInfoProps> = ({
  customerId,
  initialCustomer,
  onClose,
  onUpdate,
  accessToken,
  userRole,
  defaultTab = "overview",
}) => {
  const [customer, setCustomer] = useState<Customer | null>(initialCustomer);
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState(defaultTab);

  useEffect(() => {
    setCustomer(initialCustomer);
  }, [initialCustomer]);

  useEffect(() => {
    if (customer) {
      const op = customer.object_permission;
      form.setFieldsValue({
        ...customer,
        max_budget: customer.litellm_budget_table?.max_budget,
        budget_duration: customer.litellm_budget_table?.budget_duration,
        allowed_mcp_servers_and_groups: {
          servers: op?.mcp_servers ?? [],
          accessGroups: op?.mcp_access_groups ?? [],
        },
        allowed_agents_and_groups: {
          agents: op?.agents ?? [],
          accessGroups: op?.agent_access_groups ?? [],
        },
        mcp_tool_permissions: op?.mcp_tool_permissions ?? {},
      });
    }
  }, [customer, form]);

  const copyToClipboard = async () => {
    if (!customerId) return;
    const success = await utilCopyToClipboard(customerId);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const buildObjectPermission = (values: Record<string, any>) => {
    const objPerm: {
      mcp_servers?: string[];
      mcp_access_groups?: string[];
      mcp_tool_permissions?: Record<string, string[]>;
      agents?: string[];
      agent_access_groups?: string[];
    } = {};
    const mcp = values.allowed_mcp_servers_and_groups;
    if (mcp && (mcp.servers?.length > 0 || mcp.accessGroups?.length > 0)) {
      if (mcp.servers?.length) objPerm.mcp_servers = mcp.servers;
      if (mcp.accessGroups?.length) objPerm.mcp_access_groups = mcp.accessGroups;
    }
    if (values.mcp_tool_permissions && Object.keys(values.mcp_tool_permissions).length > 0) {
      objPerm.mcp_tool_permissions = values.mcp_tool_permissions;
    }
    const agents = values.allowed_agents_and_groups;
    if (agents && (agents.agents?.length > 0 || agents.accessGroups?.length > 0)) {
      if (agents.agents?.length) objPerm.agents = agents.agents;
      if (agents.accessGroups?.length) objPerm.agent_access_groups = agents.accessGroups;
    }
    return Object.keys(objPerm).length > 0 ? objPerm : undefined;
  };

  const handleSave = async () => {
    if (!customer || !accessToken) return;
    try {
      const values = await form.validateFields();
      setLoading(true);
      const {
        allowed_mcp_servers_and_groups,
        allowed_agents_and_groups,
        mcp_tool_permissions,
        ...rest
      } = values;
      const object_permission = buildObjectPermission(values);
      const updated: Customer = {
        ...customer,
        ...rest,
        ...(object_permission ? { object_permission } : {}),
      };
      await customerUpdateCall(accessToken, updated);
      setCustomer(updated);
      onUpdate(updated);
    } catch (error) {
      console.error("Validation failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const canEdit = userRole === "Admin" || userRole === "Org Admin";

  if (!customer && !initialCustomer) {
    return (
      <div className="p-4">
        <Button type="text" icon={<ArrowLeftIcon className="h-4 w-4" />} onClick={onClose} className="mb-4">
          Back to Customers
        </Button>
        <Text>Customer not found.</Text>
      </div>
    );
  }

  const c = customer ?? initialCustomer!;
  const displayName = c.alias || c.user_id;

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button
            type="text"
            icon={<ArrowLeftIcon className="h-4 w-4" />}
            onClick={onClose}
            className="mb-4"
          >
            Back to Customers
          </Button>
          <Title>{displayName}</Title>
          <div className="flex items-center gap-2">
            <Text className="text-gray-500 font-mono text-sm">{c.user_id}</Text>
            <Button
              type="text"
              size="small"
              icon={copied ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={copyToClipboard}
              className={copied ? "text-green-600" : "text-gray-500 hover:text-gray-700"}
            />
          </div>
        </div>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={(key) => setActiveTab(key as "overview" | "settings")}
        className="mb-4"
        items={[
          {
            key: "overview",
            label: "Overview",
            children: (
              <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
                <Card>
                  <Text>Spend (USD)</Text>
                  <div className="mt-2">
                    <Title>${formatNumberWithCommas(c.spend, 4)}</Title>
                  </div>
                </Card>
                <Card>
                  <Text>Budget (USD)</Text>
                  <div className="mt-2">
                    <Text>
                      {c.litellm_budget_table?.max_budget != null
                        ? `$${formatNumberWithCommas(c.litellm_budget_table.max_budget, 4)}`
                        : "No limit"}
                    </Text>
                    {c.litellm_budget_table?.budget_duration && (
                      <Text className="text-gray-500 block">Duration: {c.litellm_budget_table.budget_duration}</Text>
                    )}
                  </div>
                </Card>
                <Card>
                  <Text>Default Model</Text>
                  <div className="mt-2">
                    {c.default_model ? (
                      <Badge color="gray">{c.default_model}</Badge>
                    ) : (
                      <Text className="text-gray-500">—</Text>
                    )}
                  </div>
                </Card>
                <Card>
                  <Text>Region</Text>
                  <div className="mt-2">
                    <Text>{c.allowed_model_region ? c.allowed_model_region.toUpperCase() : "Any"}</Text>
                  </div>
                </Card>
                <Card>
                  <Text>Status</Text>
                  <div className="mt-2">
                    {c.blocked ? (
                      <Badge color="red">Blocked</Badge>
                    ) : (
                      <Badge color="green">Active</Badge>
                    )}
                  </div>
                </Card>

                <ObjectPermissionsView
                  objectPermission={{
                    object_permission_id: "",
                    vector_stores: [],
                    mcp_servers: c.object_permission?.mcp_servers ?? [],
                    mcp_access_groups: c.object_permission?.mcp_access_groups ?? [],
                    mcp_tool_permissions: c.object_permission?.mcp_tool_permissions ?? {},
                    agents: c.object_permission?.agents ?? [],
                    agent_access_groups: c.object_permission?.agent_access_groups ?? [],
                  }}
                  variant="card"
                  accessToken={accessToken}
                />
              </Grid>
            ),
          },
          ...(canEdit
            ? [
                {
                  key: "settings",
                  label: "Settings",
                  children: (
                    <Card className="overflow-y-auto max-h-[65vh]">
                      <div className="flex justify-between items-center mb-4">
                        <Title>Customer Settings</Title>
                      </div>
                      <Form form={form} layout="vertical" onFinish={handleSave}>
                        <Form.Item label="Customer ID">
                          <Input value={c.user_id} disabled className="font-mono bg-gray-50" />
                        </Form.Item>
                        <Form.Item label="Spend (USD)">
                          <Input value={c.spend.toFixed(4)} disabled className="bg-gray-50" />
                        </Form.Item>
                        <CustomerFormFields form={form} mode="edit" />

                        <div className="pt-6 mt-6 border-t border-gray-200">
                          <Text className="font-semibold text-gray-900 block mb-3">MCP Servers / Access Groups</Text>
                          <Form.Item
                            name="allowed_mcp_servers_and_groups"
                            help="Select MCP servers or access groups this customer can access"
                          >
                            <MCPServerSelector
                              onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
                              value={form.getFieldValue("allowed_mcp_servers_and_groups")}
                              accessToken={accessToken || ""}
                              placeholder="Select MCP servers or access groups (optional)"
                            />
                          </Form.Item>
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
                              <div className="mb-6">
                                <MCPToolPermissions
                                  accessToken={accessToken || ""}
                                  selectedServers={form.getFieldValue("allowed_mcp_servers_and_groups")?.servers || []}
                                  toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
                                  onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
                                />
                              </div>
                            )}
                          </Form.Item>
                        </div>

                        <div className="pt-6 mt-6 border-t border-gray-200">
                          <Text className="font-semibold text-gray-900 block mb-3">Agents / Access Groups</Text>
                          <Form.Item
                            name="allowed_agents_and_groups"
                            help="Select agents or access groups this customer can access"
                          >
                            <AgentSelector
                              onChange={(val: any) => form.setFieldValue("allowed_agents_and_groups", val)}
                              value={form.getFieldValue("allowed_agents_and_groups")}
                              accessToken={accessToken || ""}
                              placeholder="Select agents or access groups (optional)"
                            />
                          </Form.Item>
                        </div>

                        <Form.Item className="mt-6 mb-0">
                          <Button type="primary" htmlType="submit" loading={loading}>
                            Save Changes
                          </Button>
                        </Form.Item>
                      </Form>
                    </Card>
                  ),
                },
              ]
            : []),
        ]}
      />
    </div>
  );
};

export default CustomerInfo;

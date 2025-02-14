import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Grid,
  Badge,
  Button as TremorButton,
} from "@tremor/react";
import { Button, Form, Input, Select, message, InputNumber, Tooltip } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { Organization } from "../networking";



interface OrganizationInfoProps {
  organizationId: string;
  onClose: () => void;
  accessToken: string | null;
  is_org_admin: boolean;
  is_proxy_admin: boolean;
  userModels: string[];
  editOrg: boolean;
}

const OrganizationInfoView: React.FC<OrganizationInfoProps> = ({
  organizationId,
  onClose,
  accessToken,
  is_org_admin,
  is_proxy_admin,
  userModels,
  editOrg
}) => {
  const [orgData, setOrgData] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const [form] = Form.useForm();
  const [isEditing, setIsEditing] = useState(false);

  const canEditOrg = is_org_admin || is_proxy_admin;

  const fetchOrgInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await fetch(`/organization/info?organization_id=${organizationId}`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });
      if (!response.ok) throw new Error('Failed to fetch organization info');
      const data = await response.json();
      setOrgData(data);
    } catch (error) {
      message.error("Failed to load organization information");
      console.error("Error fetching organization info:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOrgInfo();
  }, [organizationId, accessToken]);

  const handleOrgUpdate = async (values: any) => {
    try {
      if (!accessToken) return;

      const updateData = {
        organization_id: organizationId,
        organization_alias: values.organization_alias,
        models: values.models,
        litellm_budget_table: {
          tpm_limit: values.tpm_limit,
          rpm_limit: values.rpm_limit,
          max_budget: values.max_budget,
          budget_duration: values.budget_duration,
        }
      };
      
      const response = await fetch('/organization/update', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
      });

      if (!response.ok) throw new Error('Failed to update organization');
      
      message.success("Organization settings updated successfully");
      setIsEditing(false);
      fetchOrgInfo();
    } catch (error) {
      message.error("Failed to update organization settings");
      console.error("Error updating organization:", error);
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!orgData) {
    return <div className="p-4">Organization not found</div>;
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">← Back</Button>
          <Title>{orgData.organization_alias}</Title>
          <Text className="text-gray-500 font-mono">{orgData.organization_id}</Text>
        </div>
      </div>

      <TabGroup defaultIndex={editOrg ? 2 : 0}>
        <TabList className="mb-4">
          <Tab>Overview</Tab>
          <Tab>Budget</Tab>
          <Tab>Settings</Tab>
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Organization Details</Text>
                <div className="mt-2">
                  <Text>Created: {new Date(orgData.created_at).toLocaleDateString()}</Text>
                  <Text>Updated: {new Date(orgData.updated_at).toLocaleDateString()}</Text>
                  <Text>Created By: {orgData.created_by}</Text>
                </div>
              </Card>

              <Card>
                <Text>Rate Limits</Text>
                <div className="mt-2">
                  <Text>TPM: {orgData.litellm_budget_table.tpm_limit || 'Unlimited'}</Text>
                  <Text>RPM: {orgData.litellm_budget_table.rpm_limit || 'Unlimited'}</Text>
                  {orgData.litellm_budget_table.max_parallel_requests && (
                    <Text>Max Parallel Requests: {orgData.litellm_budget_table.max_parallel_requests}</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Models</Text>
                <div className="mt-2 flex flex-wrap gap-2">
                  {orgData.models.map((model, index) => (
                    <Badge key={index} color="red">
                      {model}
                    </Badge>
                  ))}
                </div>
              </Card>
            </Grid>
          </TabPanel>

          {/* Budget Panel */}
          <TabPanel>
            <Card>
              <Title>Budget Information</Title>
              <div className="mt-4 space-y-4">
                <div>
                  <Text className="font-medium">Current Spend</Text>
                  <Title className="mt-2">${orgData.spend.toFixed(6)}</Title>
                </div>

                <div>
                  <Text className="font-medium">Budget Limits</Text>
                  <div className="mt-2">
                    <Text>Max Budget: {orgData.litellm_budget_table.max_budget === null ? "Unlimited" : `$${orgData.litellm_budget_table.max_budget}`}</Text>
                    <Text>Soft Budget: {orgData.litellm_budget_table.soft_budget === null ? "Not Set" : `$${orgData.litellm_budget_table.soft_budget}`}</Text>
                    {orgData.litellm_budget_table.budget_duration && (
                      <Text>Reset Period: {orgData.litellm_budget_table.budget_duration}</Text>
                    )}
                  </div>
                </div>

                <div>
                  <Text className="font-medium">Model Spend</Text>
                  <div className="mt-2">
                    {Object.entries(orgData.model_spend).map(([model, spend]) => (
                      <div key={model} className="flex justify-between">
                        <Text>{model}</Text>
                        <Text>${spend.toFixed(6)}</Text>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          </TabPanel>

          {/* Settings Panel */}
          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Organization Settings</Title>
                {(canEditOrg && !isEditing) && (
                  <TremorButton 
                    onClick={() => setIsEditing(true)}
                  >
                    Edit Settings
                  </TremorButton>
                )}
              </div>

              {isEditing ? (
                <Form
                  form={form}
                  onFinish={handleOrgUpdate}
                  initialValues={{
                    organization_alias: orgData.organization_alias,
                    models: orgData.models,
                    tpm_limit: orgData.litellm_budget_table.tpm_limit,
                    rpm_limit: orgData.litellm_budget_table.rpm_limit,
                    max_budget: orgData.litellm_budget_table.max_budget,
                    budget_duration: orgData.litellm_budget_table.budget_duration,
                  }}
                  layout="vertical"
                >
                  <Form.Item
                    label="Organization Name"
                    name="organization_alias"
                    rules={[{ required: true, message: "Please input an organization name" }]}
                  >
                    <Input />
                  </Form.Item>
                  
                  <Form.Item label="Models" name="models">
                    <Select
                      mode="multiple"
                      placeholder="Select models"
                    >
                      <Select.Option key="all-proxy-models" value="all-proxy-models">
                        All Proxy Models
                      </Select.Option>
                      {userModels.map((model) => (
                        <Select.Option key={model} value={model}>
                          {getModelDisplayName(model)}
                        </Select.Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item label="Max Budget (USD)" name="max_budget">
                    <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="Reset Budget" name="budget_duration">
                    <Select placeholder="n/a">
                      <Select.Option value="24h">daily</Select.Option>
                      <Select.Option value="7d">weekly</Select.Option>
                      <Select.Option value="30d">monthly</Select.Option>
                    </Select>
                  </Form.Item>

                  <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
                    <InputNumber step={1} style={{ width: "100%" }} />
                  </Form.Item>

                  <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
                    <InputNumber step={1} style={{ width: "100%" }} />
                  </Form.Item>

                  <div className="flex justify-end gap-2 mt-6">
                    <Button onClick={() => setIsEditing(false)}>
                      Cancel
                    </Button>
                    <TremorButton type="submit">
                      Save Changes
                    </TremorButton>
                  </div>
                </Form>
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Organization Name</Text>
                    <div>{orgData.organization_alias}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Organization ID</Text>
                    <div className="font-mono">{orgData.organization_id}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Created At</Text>
                    <div>{new Date(orgData.created_at).toLocaleString()}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {orgData.models.map((model, index) => (
                        <Badge key={index} color="red">
                          {model}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <Text className="font-medium">Rate Limits</Text>
                    <div>TPM: {orgData.litellm_budget_table.tpm_limit || 'Unlimited'}</div>
                    <div>RPM: {orgData.litellm_budget_table.rpm_limit || 'Unlimited'}</div>
                  </div>
                  <div>
                    <Text className="font-medium">Budget</Text>
                    <div>Max: {orgData.litellm_budget_table.max_budget !== null ? `$${orgData.litellm_budget_table.max_budget}` : 'No Limit'}</div>
                    <div>Reset: {orgData.litellm_budget_table.budget_duration || 'Never'}</div>
                  </div>
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default OrganizationInfoView;
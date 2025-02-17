import React, { useState, useEffect } from 'react';
import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
  Card,
  Text,
  Badge,
  Icon,
  Grid,
  Col,
  Button,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
} from "@tremor/react";
import { Modal, Form, InputNumber, Tooltip, Select as Select2 } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { PencilAltIcon, TrashIcon, RefreshIcon } from "@heroicons/react/outline";
import { TextInput } from "@tremor/react";
import { getModelDisplayName } from './key_team_helpers/fetch_available_models_team_key';
import { message } from 'antd';
import OrganizationInfoView from './organization/organization_view';
import { Organization, organizationListCall, organizationCreateCall, organizationDeleteCall } from './networking';
interface OrganizationsTableProps {
  organizations: Organization[];
  userRole: string;
  userModels: string[];
  accessToken: string | null;
  lastRefreshed?: string;
  handleRefreshClick?: () => void;
  currentOrg?: any;
  guardrailsList?: string[];
  setOrganizations: (organizations: Organization[]) => void;
  premiumUser: boolean;
}

export const fetchOrganizations = async (accessToken: string, setOrganizations: (organizations: Organization[]) => void) => {
  const organizations = await organizationListCall(accessToken);
  setOrganizations(organizations);
};

const OrganizationsTable: React.FC<OrganizationsTableProps> = ({
  organizations,
  userRole,
  userModels,
  accessToken,
  lastRefreshed,
  handleRefreshClick,
  currentOrg,
  guardrailsList = [],
  setOrganizations,
  premiumUser
}) => {
  const [selectedOrgId, setSelectedOrgId] = useState<string | null>(null);
  const [editOrg, setEditOrg] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [orgToDelete, setOrgToDelete] = useState<string | null>(null);
  const [isOrgModalVisible, setIsOrgModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    if (organizations.length === 0 && accessToken) {
      fetchOrganizations(accessToken, setOrganizations);
    }
  }, [organizations, accessToken]);

  const handleDelete = (orgId: string | null) => {
    if (!orgId) return;
    
    setOrgToDelete(orgId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!orgToDelete || !accessToken) return;

    try {
      await organizationDeleteCall(accessToken, orgToDelete);
      message.success('Organization deleted successfully');

      setIsDeleteModalOpen(false);
      setOrgToDelete(null);
      // Refresh organizations list
      fetchOrganizations(accessToken, setOrganizations);
    } catch (error) {
      console.error('Error deleting organization:', error);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setOrgToDelete(null);
  };

  const handleCreate = async (values: any) => {
    try {
      if (!accessToken) return;

      await organizationCreateCall(accessToken, values);
      setIsOrgModalVisible(false);
      form.resetFields();
      // Refresh organizations list
      fetchOrganizations(accessToken, setOrganizations);
    } catch (error) {
      console.error('Error creating organization:', error);
    }
  };

  const handleCancel = () => {
    setIsOrgModalVisible(false);
    form.resetFields();
  };

  if (!premiumUser) {
    return (
      <div>
        <Text>This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key <a href="https://litellm.ai/pricing" target="_blank" rel="noopener noreferrer">here</a>.</Text>
      </div>
    );
  }

  if (selectedOrgId) {
    return (
      <OrganizationInfoView
        organizationId={selectedOrgId}
        onClose={() => {
          setSelectedOrgId(null);
          setEditOrg(false);
        }}
        accessToken={accessToken}
        is_org_admin={true} // You'll need to implement proper org admin check
        is_proxy_admin={userRole === "Admin"}
        userModels={userModels}
        editOrg={editOrg}
      />
    );
  }

  return (
    <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
      <TabList className="flex justify-between mt-2 w-full items-center">
        <div className="flex">
          <Tab>Your Organizations</Tab>
        </div>
        <div className="flex items-center space-x-2">
          {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
          <Icon
            icon={RefreshIcon}
            variant="shadow"
            size="xs"
            className="self-center"
            onClick={handleRefreshClick}
          />
        </div>
      </TabList>
      <TabPanels>
        <TabPanel>
          <Text>
            Click on &ldquo;Organization ID&rdquo; to view organization details.
          </Text>
          <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
            <Col numColSpan={1}>
              <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Organization ID</TableHeaderCell>
                      <TableHeaderCell>Organization Name</TableHeaderCell>
                      <TableHeaderCell>Created</TableHeaderCell>
                      <TableHeaderCell>Spend (USD)</TableHeaderCell>
                      <TableHeaderCell>Budget (USD)</TableHeaderCell>
                      <TableHeaderCell>Models</TableHeaderCell>
                      <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
                      <TableHeaderCell>Info</TableHeaderCell>
                      <TableHeaderCell>Actions</TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    {organizations && organizations.length > 0
                      ? organizations
                          .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
                          .map((org: Organization) => (
                            <TableRow key={org.organization_id}>
                              <TableCell>
                                <div className="overflow-hidden">
                                  <Tooltip title={org.organization_id}>
                                    <Button
                                      size="xs"
                                      variant="light"
                                      className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
                                      onClick={() => setSelectedOrgId(org.organization_id)}
                                    >
                                      {org.organization_id?.slice(0, 7)}...
                                    </Button>
                                  </Tooltip>
                                </div>
                              </TableCell>
                              <TableCell>{org.organization_alias}</TableCell>
                              <TableCell>
                                {org.created_at ? new Date(org.created_at).toLocaleDateString() : "N/A"}
                              </TableCell>
                              <TableCell>{org.spend}</TableCell>
                              <TableCell>
                                {org.litellm_budget_table?.max_budget !== null && org.litellm_budget_table?.max_budget !== undefined ? org.litellm_budget_table?.max_budget : "No limit"}
                              </TableCell>
                              <TableCell>
                                {Array.isArray(org.models) && (
                                  <div className="flex flex-col">
                                    {org.models.length === 0 ? (
                                      <Badge size="xs" className="mb-1" color="red">
                                        All Proxy Models
                                      </Badge>
                                    ) : (
                                      org.models.map((model, index) =>
                                        model === "all-proxy-models" ? (
                                          <Badge key={index} size="xs" className="mb-1" color="red">
                                            All Proxy Models
                                          </Badge>
                                        ) : (
                                          <Badge key={index} size="xs" className="mb-1" color="blue">
                                            {model.length > 30
                                              ? `${getModelDisplayName(model).slice(0, 30)}...`
                                              : getModelDisplayName(model)}
                                          </Badge>
                                        )
                                      )
                                    )}
                                  </div>
                                )}
                              </TableCell>
                              <TableCell>
                                <Text>
                                  TPM: {org.litellm_budget_table?.tpm_limit ? org.litellm_budget_table?.tpm_limit : "Unlimited"}
                                  <br />
                                  RPM: {org.litellm_budget_table?.rpm_limit ? org.litellm_budget_table?.rpm_limit : "Unlimited"}
                                </Text>
                              </TableCell>
                              <TableCell>
                                <Text>{org.members?.length || 0} Members</Text>
                              </TableCell>
                              <TableCell>
                                {userRole === "Admin" && (
                                  <>
                                    <Icon
                                      icon={PencilAltIcon}
                                      size="sm"
                                      onClick={() => {
                                        setSelectedOrgId(org.organization_id);
                                        setEditOrg(true);
                                      }}
                                    />
                                    <Icon
                                      onClick={() => handleDelete(org.organization_id)}
                                      icon={TrashIcon}
                                      size="sm"
                                    />
                                  </>
                                )}
                              </TableCell>
                            </TableRow>
                          ))
                      : null}
                  </TableBody>
                </Table>
              </Card>
            </Col>
            {(userRole === "Admin" || userRole === "Org Admin") && (
              <Col numColSpan={1}>
                <Button
                  className="mx-auto"
                  onClick={() => setIsOrgModalVisible(true)}
                >
                  + Create New Organization
                </Button>
                <Modal
                  title="Create Organization"
                  visible={isOrgModalVisible}
                  width={800}
                  footer={null}
                  onCancel={handleCancel}
                >
                  <Form
                    form={form}
                    onFinish={handleCreate}
                    labelCol={{ span: 8 }}
                    wrapperCol={{ span: 16 }}
                    labelAlign="left"
                  >
                    <Form.Item
                      label="Organization Name"
                      name="organization_alias"
                      rules={[{ required: true, message: "Please input an organization name" }]}
                    >
                      <TextInput placeholder="" />
                    </Form.Item>
                    <Form.Item label="Models" name="models">
                      <Select2
                        mode="multiple"
                        placeholder="Select models"
                        style={{ width: "100%" }}
                      >
                        <Select2.Option key="all-proxy-models" value="all-proxy-models">
                          All Proxy Models
                        </Select2.Option>
                        {userModels && userModels.length > 0 && userModels.map((model) => (
                          <Select2.Option key={model} value={model}>
                            {getModelDisplayName(model)}
                          </Select2.Option>
                        ))}
                      </Select2>
                    </Form.Item>

                    <Form.Item label="Max Budget (USD)" name="max_budget">
                      <InputNumber step={0.01} precision={2} width={200} />
                    </Form.Item>
                    <Form.Item label="Reset Budget" name="budget_duration">
                      <Select2 defaultValue={null} placeholder="n/a">
                        <Select2.Option value="24h">daily</Select2.Option>
                        <Select2.Option value="7d">weekly</Select2.Option>
                        <Select2.Option value="30d">monthly</Select2.Option>
                      </Select2>
                    </Form.Item>
                    <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
                      <InputNumber step={1} width={400} />
                    </Form.Item>
                    <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
                      <InputNumber step={1} width={400} />
                    </Form.Item>

                    <div style={{ textAlign: "right", marginTop: "10px" }}>
                      <Button type="submit">Create Organization</Button>
                    </div>
                  </Form>
                </Modal>
              </Col>
            )}
          </Grid>
        </TabPanel>
      </TabPanels>

      {isDeleteModalOpen ? (
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
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Delete Organization
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to delete this organization?
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button onClick={confirmDelete} color="red" className="ml-2">
                  Delete
                </Button>
                <Button onClick={cancelDelete}>Cancel</Button>
              </div>
            </div>
          </div>
        </div>
      ) : <></>}
    </TabGroup>
  );
};

export default OrganizationsTable;
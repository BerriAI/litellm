import { organizationKeys, useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useUserModels } from "@/app/(dashboard)/hooks/models/useModels";
import OrganizationFilters, { FilterState } from "@/app/(dashboard)/organizations/OrganizationFilters";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Form, Input, Modal, Select as Select2, Tooltip } from "antd";
import { useQueryClient } from "@tanstack/react-query";
import React, { useState } from "react";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { organizationCreateCall, organizationDeleteCall } from "@/components/networking";
import OrganizationInfoView from "@/components/organization/organization_view";
import NumericalInput from "@/components/shared/numerical_input";
import { Button } from "@/components/ui/button";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";

import OrganizationsTable from "./OrganizationsTable";

interface OrganizationsPanelProps {
  userRole: string;
  accessToken: string | null;
  premiumUser: boolean;
}

const OrganizationsPanel: React.FC<OrganizationsPanelProps> = ({ userRole, accessToken, premiumUser }) => {
  const [selectedOrgId, setSelectedOrgId] = useState<string | null>(null);
  const [editOrg, setEditOrg] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [orgToDelete, setOrgToDelete] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isOrgModalVisible, setIsOrgModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({ org_id: "", org_alias: "" });

  const queryClient = useQueryClient();
  const { data: organizations = [], isLoading } = useOrganizations({
    org_id: filters.org_id,
    org_alias: filters.org_alias,
  });
  const { data: userModels = [] } = useUserModels();

  const searchActive = Boolean(filters.org_id || filters.org_alias);

  const refetchOrganizations = () => queryClient.invalidateQueries({ queryKey: organizationKeys.lists() });

  const handleFilterChange = (key: keyof FilterState, value: string) => {
    setFilters((previousFilters) => ({ ...previousFilters, [key]: value }));
  };

  const handleFilterReset = () => {
    setFilters({ org_id: "", org_alias: "" });
  };

  const handleDelete = (orgId: string | null) => {
    if (!orgId) return;

    setOrgToDelete(orgId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!orgToDelete || !accessToken) return;

    try {
      setIsDeleting(true);
      await organizationDeleteCall(accessToken, orgToDelete);
      NotificationsManager.success("Organization deleted successfully");

      setIsDeleteModalOpen(false);
      setOrgToDelete(null);
      await refetchOrganizations();
    } catch (error) {
      console.error("Error deleting organization:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setOrgToDelete(null);
  };

  const handleCreate = async (values: any) => {
    try {
      if (!accessToken) return;

      // Transform allowed_vector_store_ids and allowed_mcp_servers_and_groups into object_permission
      if (
        (values.allowed_vector_store_ids && values.allowed_vector_store_ids.length > 0) ||
        (values.allowed_mcp_servers_and_groups &&
          (values.allowed_mcp_servers_and_groups.servers?.length > 0 ||
            values.allowed_mcp_servers_and_groups.accessGroups?.length > 0))
      ) {
        values.object_permission = {};
        if (values.allowed_vector_store_ids && values.allowed_vector_store_ids.length > 0) {
          values.object_permission.vector_stores = values.allowed_vector_store_ids;
          delete values.allowed_vector_store_ids;
        }
        if (values.allowed_mcp_servers_and_groups) {
          if (values.allowed_mcp_servers_and_groups.servers?.length > 0) {
            values.object_permission.mcp_servers = values.allowed_mcp_servers_and_groups.servers;
          }
          if (values.allowed_mcp_servers_and_groups.accessGroups?.length > 0) {
            values.object_permission.mcp_access_groups = values.allowed_mcp_servers_and_groups.accessGroups;
          }
          delete values.allowed_mcp_servers_and_groups;
        }
      }

      await organizationCreateCall(accessToken, values);
      NotificationsManager.success("Organization created successfully");
      setIsOrgModalVisible(false);
      form.resetFields();
      await refetchOrganizations();
    } catch (error) {
      console.error("Error creating organization:", error);
    }
  };

  const handleCancel = () => {
    setIsOrgModalVisible(false);
    form.resetFields();
  };

  if (!premiumUser) {
    return (
      <div className="mx-4 mt-4">
        <p className="text-sm text-muted-foreground">
          This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key{" "}
          <a
            href="https://www.litellm.ai/#pricing"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline-offset-4 hover:underline"
          >
            here
          </a>
          .
        </p>
      </div>
    );
  }

  return (
    <div className="mx-4 mt-4 flex flex-col gap-4">
      {(userRole === "Admin" || userRole === "Org Admin") && (
        <Button className="w-fit" onClick={() => setIsOrgModalVisible(true)}>
          + Create New Organization
        </Button>
      )}

      {selectedOrgId ? (
        <OrganizationInfoView
          organizationId={selectedOrgId}
          onClose={() => {
            setSelectedOrgId(null);
            setEditOrg(false);
          }}
          accessToken={accessToken}
          is_org_admin={true}
          is_proxy_admin={userRole === "Admin"}
          userModels={userModels}
          editOrg={editOrg}
        />
      ) : (
        <>
          <p className="text-sm text-muted-foreground">Click on an organization ID to view its details.</p>
          <OrganizationFilters
            filters={filters}
            showFilters={showFilters}
            onToggleFilters={setShowFilters}
            onChange={handleFilterChange}
            onReset={handleFilterReset}
          />
          <OrganizationsTable
            organizations={organizations}
            isLoading={isLoading}
            userRole={userRole}
            searchActive={searchActive}
            onOrganizationClick={setSelectedOrgId}
            onEditClick={(organizationId) => {
              setSelectedOrgId(organizationId);
              setEditOrg(true);
            }}
            onDeleteClick={handleDelete}
          />
        </>
      )}

      <Modal title="Create Organization" visible={isOrgModalVisible} width={800} footer={null} onCancel={handleCancel}>
        <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
          <Form.Item
            label="Organization Name"
            name="organization_alias"
            rules={[
              {
                required: true,
                message: "Please input an organization name",
              },
            ]}
          >
            <Input placeholder="" />
          </Form.Item>
          <Form.Item label="Models" name="models">
            <ModelSelect
              options={{ showAllProxyModelsOverride: true, includeSpecialOptions: true }}
              value={form.getFieldValue("models")}
              onChange={(values) => form.setFieldValue("models", values)}
              context="organization"
            />
          </Form.Item>

          <Form.Item label="Max Budget (USD)" name="max_budget">
            <NumericalInput step={0.01} precision={2} width={200} />
          </Form.Item>
          <Form.Item label="Reset Budget" name="budget_duration">
            <Select2 defaultValue={null} placeholder="n/a">
              <Select2.Option value="24h">daily</Select2.Option>
              <Select2.Option value="7d">weekly</Select2.Option>
              <Select2.Option value="30d">monthly</Select2.Option>
            </Select2>
          </Form.Item>
          <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
            <NumericalInput step={1} width={400} />
          </Form.Item>
          <Form.Item label="Requests per minute Limit (RPM)" name="rpm_limit">
            <NumericalInput step={1} width={400} />
          </Form.Item>

          <Form.Item
            label={
              <span>
                Allowed Vector Stores{" "}
                <Tooltip title="Select which vector stores this organization can access by default. Leave empty for access to all vector stores">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="allowed_vector_store_ids"
            className="mt-4"
            help="Select vector stores this organization can access. Leave empty for access to all vector stores"
          >
            <VectorStoreSelector
              onChange={(values) => form.setFieldValue("allowed_vector_store_ids", values)}
              value={form.getFieldValue("allowed_vector_store_ids")}
              accessToken={accessToken || ""}
              placeholder="Select vector stores (optional)"
            />
          </Form.Item>

          <Form.Item
            label={
              <span>
                Allowed MCP Servers{" "}
                <Tooltip title="Select which MCP servers and access groups this organization can access by default.">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="allowed_mcp_servers_and_groups"
            className="mt-4"
            help="Select MCP servers and access groups this organization can access."
          >
            <MCPServerSelector
              onChange={(values) => form.setFieldValue("allowed_mcp_servers_and_groups", values)}
              value={form.getFieldValue("allowed_mcp_servers_and_groups")}
              accessToken={accessToken || ""}
              placeholder="Select MCP servers and access groups (optional)"
            />
          </Form.Item>

          <Form.Item label="Metadata" name="metadata">
            <Input.TextArea rows={4} />
          </Form.Item>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button type="submit">Create Organization</Button>
          </div>
        </Form>
      </Modal>

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Organization?"
        message="Are you sure you want to delete this organization? This action cannot be undone."
        resourceInformationTitle="Organization Information"
        resourceInformation={[{ label: "Organization ID", value: orgToDelete, code: true }]}
        onCancel={cancelDelete}
        onOk={confirmDelete}
        confirmLoading={isDeleting}
      />
    </div>
  );
};

export default OrganizationsPanel;

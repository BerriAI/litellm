/**
 * Component to allow proxy admin to grant access to requested public access models
 */
import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Button,
  Badge,
  Text,
  Callout,
} from "@tremor/react";
import { Modal, Tooltip } from "antd";
import { CheckCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { getRequestedPublicAccessModels, grantRequestedPublicAccess } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { all_admin_roles } from "@/utils/roles";

interface ModelInfo {
  id: string;
  db_model: boolean;
  updated_at: string;
  updated_by: string;
  created_at: string;
  created_by: string;
  team_id: string;
  team_public_model_name: string;
  has_requested_public_access: string;
}

interface LiteLLMParams {
  custom_llm_provider: string;
  use_in_pass_through: boolean;
  use_litellm_proxy: boolean;
  merge_reasoning_content_in_choices: boolean;
  model: string;
}

interface RequestedPublicAccessModel {
  model_name: string;
  litellm_params: LiteLLMParams;
  model_info: ModelInfo;
}

interface PublicAccessRequestsProps {
  accessToken: string | null;
  userRole?: string | null;
}

const PublicAccessRequests: React.FC<PublicAccessRequestsProps> = ({ accessToken, userRole }) => {
  const [requestedModels, setRequestedModels] = useState<RequestedPublicAccessModel[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<RequestedPublicAccessModel | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [grantingAccess, setGrantingAccess] = useState(false);

  const fetchRequestedModels = async () => {
    if (!accessToken) {
      NotificationsManager.error("No access token available");
      return;
    }

    try {
      setLoading(true);
      const data = await getRequestedPublicAccessModels(accessToken);
      console.log("Requested public access models:", data);
      setRequestedModels(data || []);
    } catch (error) {
      console.error("Error fetching requested models:", error);
      NotificationsManager.error("Failed to fetch requested public access models");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const isAdmin = userRole ? all_admin_roles.includes(userRole) : false;
    if (isAdmin) {
      fetchRequestedModels();
    }
  }, [accessToken, userRole]);

  const handleGrantAccess = async () => {
    if (!accessToken || !selectedModel) {
      return;
    }

    try {
      setGrantingAccess(true);
      await grantRequestedPublicAccess(accessToken, selectedModel.model_info.id);
      NotificationsManager.success(
        `Successfully granted public access to ${selectedModel.model_info.team_public_model_name || selectedModel.model_name}`,
      );
      setIsModalVisible(false);
      setSelectedModel(null);
      // Refresh the list
      await fetchRequestedModels();
    } catch (error) {
      console.error("Error granting access:", error);
      NotificationsManager.error("Failed to grant public access");
    } finally {
      setGrantingAccess(false);
    }
  };

  const showGrantModal = (model: RequestedPublicAccessModel) => {
    setSelectedModel(model);
    setIsModalVisible(true);
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const isAdmin = userRole ? all_admin_roles.includes(userRole) : false;

  if (!isAdmin) {
    return (
      <div className="w-full">
        <Card>
          <Title>Public Access Requests</Title>
          <Callout title="Admin Access Required" color="red" className="mt-4">
            Only proxy administrators can view and manage public access requests.
          </Callout>
        </Card>
      </div>
    );
  }

  return (
    <div className="w-full">
      <Card>
        <div className="flex items-center justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Title>Public Access Requests</Title>
              <Badge color="blue" size="sm">Beta</Badge>
            </div>
            <Subtitle>Review and approve model public access requests from teams</Subtitle>
            <Text className="text-sm text-gray-600 mt-2">
              This feature allows team admins to request making their models publicly accessible. This is particularly 
              useful for teams building Bedrock agents or custom models that other teams need to consume. Once approved, 
              these models become available in the public model hub for all teams to use.
            </Text>
          </div>
          <Button onClick={fetchRequestedModels} loading={loading} variant="secondary" size="sm" className="ml-4">
            Refresh
          </Button>
        </div>

        {loading && requestedModels.length === 0 ? (
          <div className="text-center py-8">
            <Text>Loading requested models...</Text>
          </div>
        ) : requestedModels.length === 0 ? (
          <div className="text-center py-8">
            <Text>No pending public access requests</Text>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell className="w-48">Public Model Name</TableHeaderCell>
                  <TableHeaderCell className="w-32">Provider</TableHeaderCell>
                  <TableHeaderCell className="w-64">Model</TableHeaderCell>
                  <TableHeaderCell className="w-40">Team ID</TableHeaderCell>
                  <TableHeaderCell className="w-28">Status</TableHeaderCell>
                  <TableHeaderCell className="w-44">Requested Date</TableHeaderCell>
                  <TableHeaderCell className="w-36">Actions</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {requestedModels.map((model) => (
                  <TableRow key={model.model_info.id}>
                    <TableCell className="w-48">
                      <div className="flex items-center gap-2">
                        <Tooltip title={model.model_info.team_public_model_name || model.model_name}>
                          <Text className="font-medium truncate max-w-[160px] cursor-help">
                            {model.model_info.team_public_model_name || model.model_name}
                          </Text>
                        </Tooltip>
                        <Tooltip title="This is the name that will be publicly accessible">
                          <InfoCircleOutlined className="text-gray-400 flex-shrink-0" />
                        </Tooltip>
                      </div>
                    </TableCell>
                    <TableCell className="w-32">
                      <Badge color="blue">{model.litellm_params.custom_llm_provider}</Badge>
                    </TableCell>
                    <TableCell className="w-64">
                      <Tooltip title={model.litellm_params.model}>
                        <Text className="font-mono text-xs truncate block max-w-[240px]">
                          {model.litellm_params.model}
                        </Text>
                      </Tooltip>
                    </TableCell>
                    <TableCell className="w-40">
                      <Tooltip title={model.model_info.team_id}>
                        <Text className="font-mono text-xs truncate block max-w-[144px]">
                          {model.model_info.team_id}
                        </Text>
                      </Tooltip>
                    </TableCell>
                    <TableCell className="w-28">
                      <Badge color="yellow">{model.model_info.has_requested_public_access}</Badge>
                    </TableCell>
                    <TableCell className="w-44">
                      <Text className="text-xs">{formatDate(model.model_info.created_at)}</Text>
                    </TableCell>
                    <TableCell className="w-36">
                      <Button
                        size="xs"
                        color="green"
                        icon={CheckCircleOutlined}
                        onClick={() => showGrantModal(model)}
                      >
                        Grant Access
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Card>

      <Modal
        title="Grant Public Access"
        open={isModalVisible}
        onOk={handleGrantAccess}
        onCancel={() => {
          setIsModalVisible(false);
          setSelectedModel(null);
        }}
        okText="Grant Access"
        okButtonProps={{ 
          loading: grantingAccess, 
          type: "primary",
          style: { backgroundColor: '#1890ff', borderColor: '#1890ff' }
        }}
        cancelButtonProps={{ disabled: grantingAccess }}
      >
        {selectedModel && (
          <div className="space-y-4">
            <p>
              Are you sure you want to grant public access to this model deployment? Once granted, you will be able to grant access to user/team/key access.
            </p>

            <div className="bg-gray-50 p-4 rounded-md space-y-2">
              <div>
                <Text className="font-semibold">Public Name:</Text>
                <Text className="ml-2">
                  {selectedModel.model_info.team_public_model_name || selectedModel.model_name}
                </Text>
              </div>
              <div>
                <Text className="font-semibold">Provider:</Text>
                <Text className="ml-2">{selectedModel.litellm_params.custom_llm_provider}</Text>
              </div>
              <div>
                <Text className="font-semibold">Model:</Text>
                <Text className="ml-2 font-mono text-xs">{selectedModel.litellm_params.model}</Text>
              </div>
              <div>
                <Text className="font-semibold">Team ID:</Text>
                <Text className="ml-2 font-mono text-xs">{selectedModel.model_info.team_id}</Text>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default PublicAccessRequests;


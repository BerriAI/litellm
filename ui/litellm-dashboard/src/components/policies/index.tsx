import React, { useState, useEffect, useCallback } from "react";
import { Button, Tabs, Modal, message } from "antd";
import { PlusOutlined, ExclamationCircleOutlined } from "@ant-design/icons";
import { isAdminRole } from "@/utils/roles";
import PolicyTable from "./policy_table";
import PolicyInfoView from "./policy_info";
import AddPolicyForm from "./add_policy_form";
import AttachmentTable from "./attachment_table";
import AddAttachmentForm from "./add_attachment_form";
import {
  Policy,
  PolicyAttachment,
  PolicyCreateRequest,
  PolicyUpdateRequest,
  PolicyAttachmentCreateRequest,
} from "./types";
import { Guardrail } from "../guardrails/types";

interface PoliciesPanelProps {
  accessToken: string | null;
  userRole?: string;
  // API functions passed from parent
  getPoliciesList: (accessToken: string) => Promise<{ policies: Policy[]; total_count: number }>;
  createPolicy: (accessToken: string, data: PolicyCreateRequest) => Promise<Policy>;
  updatePolicy: (accessToken: string, policyId: string, data: PolicyUpdateRequest) => Promise<Policy>;
  deletePolicy: (accessToken: string, policyId: string) => Promise<void>;
  getPolicy: (accessToken: string, policyId: string) => Promise<Policy>;
  getAttachmentsList: (accessToken: string) => Promise<{ attachments: PolicyAttachment[]; total_count: number }>;
  createAttachment: (accessToken: string, data: PolicyAttachmentCreateRequest) => Promise<PolicyAttachment>;
  deleteAttachment: (accessToken: string, attachmentId: string) => Promise<void>;
  getGuardrailsList: (accessToken: string) => Promise<{ guardrails: Guardrail[] }>;
}

const PoliciesPanel: React.FC<PoliciesPanelProps> = ({
  accessToken,
  userRole,
  getPoliciesList,
  createPolicy,
  updatePolicy,
  deletePolicy,
  getPolicy,
  getAttachmentsList,
  createAttachment,
  deleteAttachment,
  getGuardrailsList,
}) => {
  const [policiesList, setPoliciesList] = useState<Policy[]>([]);
  const [attachmentsList, setAttachmentsList] = useState<PolicyAttachment[]>([]);
  const [guardrailsList, setGuardrailsList] = useState<Guardrail[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAttachmentsLoading, setIsAttachmentsLoading] = useState(false);
  const [isAddPolicyModalVisible, setIsAddPolicyModalVisible] = useState(false);
  const [isAddAttachmentModalVisible, setIsAddAttachmentModalVisible] = useState(false);
  const [editingPolicy, setEditingPolicy] = useState<Policy | null>(null);
  const [selectedPolicyId, setSelectedPolicyId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("policies");

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchPolicies = useCallback(async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const response = await getPoliciesList(accessToken);
      setPoliciesList(response.policies);
    } catch (error) {
      console.error("Error fetching policies:", error);
      message.error("Failed to fetch policies");
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, getPoliciesList]);

  const fetchAttachments = useCallback(async () => {
    if (!accessToken) return;

    setIsAttachmentsLoading(true);
    try {
      const response = await getAttachmentsList(accessToken);
      setAttachmentsList(response.attachments);
    } catch (error) {
      console.error("Error fetching attachments:", error);
      message.error("Failed to fetch attachments");
    } finally {
      setIsAttachmentsLoading(false);
    }
  }, [accessToken, getAttachmentsList]);

  const fetchGuardrails = useCallback(async () => {
    if (!accessToken) return;

    try {
      const response = await getGuardrailsList(accessToken);
      setGuardrailsList(response.guardrails);
    } catch (error) {
      console.error("Error fetching guardrails:", error);
    }
  }, [accessToken, getGuardrailsList]);

  useEffect(() => {
    fetchPolicies();
    fetchAttachments();
    fetchGuardrails();
  }, [fetchPolicies, fetchAttachments, fetchGuardrails]);

  const handleCreatePolicy = async (data: PolicyCreateRequest) => {
    if (!accessToken) return;
    await createPolicy(accessToken, data);
    message.success("Policy created successfully");
  };

  const handleUpdatePolicy = async (policyId: string, data: PolicyUpdateRequest) => {
    if (!accessToken) return;
    await updatePolicy(accessToken, policyId, data);
    message.success("Policy updated successfully");
  };

  const handleDeletePolicy = (policyId: string, policyName: string) => {
    Modal.confirm({
      title: "Delete Policy",
      icon: <ExclamationCircleOutlined />,
      content: `Are you sure you want to delete the policy "${policyName}"? This action cannot be undone.`,
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        if (!accessToken) return;
        try {
          await deletePolicy(accessToken, policyId);
          message.success("Policy deleted successfully");
          fetchPolicies();
        } catch (error) {
          console.error("Error deleting policy:", error);
          message.error("Failed to delete policy");
        }
      },
    });
  };

  const handleCreateAttachment = async (data: PolicyAttachmentCreateRequest) => {
    if (!accessToken) return;
    await createAttachment(accessToken, data);
    message.success("Attachment created successfully");
  };

  const handleDeleteAttachment = (attachmentId: string) => {
    Modal.confirm({
      title: "Delete Attachment",
      icon: <ExclamationCircleOutlined />,
      content: "Are you sure you want to delete this attachment? This action cannot be undone.",
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        if (!accessToken) return;
        try {
          await deleteAttachment(accessToken, attachmentId);
          message.success("Attachment deleted successfully");
          fetchAttachments();
        } catch (error) {
          console.error("Error deleting attachment:", error);
          message.error("Failed to delete attachment");
        }
      },
    });
  };

  const fetchPolicyById = useCallback(
    async (policyId: string): Promise<Policy | null> => {
      if (!accessToken) return null;
      try {
        return await getPolicy(accessToken, policyId);
      } catch (error) {
        console.error("Error fetching policy:", error);
        return null;
      }
    },
    [accessToken, getPolicy]
  );

  const handlePolicySuccess = () => {
    fetchPolicies();
    setEditingPolicy(null);
  };

  const handleAttachmentSuccess = () => {
    fetchAttachments();
  };

  const tabItems = [
    {
      key: "policies",
      label: "Policies",
      children: selectedPolicyId ? (
        <PolicyInfoView
          policyId={selectedPolicyId}
          onClose={() => setSelectedPolicyId(null)}
          onEdit={(policy) => {
            setEditingPolicy(policy);
            setIsAddPolicyModalVisible(true);
            setSelectedPolicyId(null);
          }}
          accessToken={accessToken}
          isAdmin={isAdmin}
          fetchPolicy={fetchPolicyById}
        />
      ) : (
        <>
          <div style={{ marginBottom: 16 }}>
            {isAdmin && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => {
                  setEditingPolicy(null);
                  setIsAddPolicyModalVisible(true);
                }}
              >
                Add New Policy
              </Button>
            )}
          </div>
          <PolicyTable
            policies={policiesList}
            isLoading={isLoading}
            onDeleteClick={handleDeletePolicy}
            onEditClick={(policy) => {
              setEditingPolicy(policy);
              setIsAddPolicyModalVisible(true);
            }}
            onViewClick={(policyId) => setSelectedPolicyId(policyId)}
            isAdmin={isAdmin}
          />
        </>
      ),
    },
    {
      key: "attachments",
      label: "Attachments",
      children: (
        <>
          <div style={{ marginBottom: 16 }}>
            {isAdmin && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setIsAddAttachmentModalVisible(true)}
                disabled={policiesList.length === 0}
              >
                Add New Attachment
              </Button>
            )}
          </div>
          <AttachmentTable
            attachments={attachmentsList}
            isLoading={isAttachmentsLoading}
            onDeleteClick={handleDeleteAttachment}
            isAdmin={isAdmin}
          />
        </>
      ),
    },
  ];

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
      />

      <AddPolicyForm
        visible={isAddPolicyModalVisible}
        onClose={() => {
          setIsAddPolicyModalVisible(false);
          setEditingPolicy(null);
        }}
        onSuccess={handlePolicySuccess}
        accessToken={accessToken}
        editingPolicy={editingPolicy}
        existingPolicies={policiesList}
        availableGuardrails={guardrailsList}
        onCreatePolicy={handleCreatePolicy}
        onUpdatePolicy={handleUpdatePolicy}
      />

      <AddAttachmentForm
        visible={isAddAttachmentModalVisible}
        onClose={() => setIsAddAttachmentModalVisible(false)}
        onSuccess={handleAttachmentSuccess}
        accessToken={accessToken}
        policies={policiesList}
        onCreateAttachment={handleCreateAttachment}
      />
    </div>
  );
};

export default PoliciesPanel;

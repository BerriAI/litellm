import React, { useState, useEffect, useCallback } from "react";
import { Button, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { Modal, message, Alert } from "antd";
import { ExclamationCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { isAdminRole } from "@/utils/roles";
import PolicyTable from "./policy_table";
import PolicyInfoView from "./policy_info";
import AddPolicyForm from "./add_policy_form";
import { FlowBuilderPage } from "./pipeline_flow_builder";
import AttachmentTable from "./attachment_table";
import AddAttachmentForm from "./add_attachment_form";
import PolicyTestPanel from "./policy_test_panel";
import PolicyTemplates from "./policy_templates";
import GuardrailSelectionModal from "./guardrail_selection_modal";
import TemplateParameterModal from "./template_parameter_modal";
import AiSuggestionModal from "./ai_suggestion_modal";
import {
  getPoliciesList,
  deletePolicyCall,
  getPolicyAttachmentsList,
  deletePolicyAttachmentCall,
  getGuardrailsList,
  getPolicyInfo,
  createPolicyCall,
  updatePolicyCall,
  createPolicyAttachmentCall,
  createGuardrailCall,
  enrichPolicyTemplate,
} from "../networking";
import {
  Policy,
  PolicyAttachment,
} from "./types";
import { Guardrail } from "../guardrails/types";
import DeleteResourceModal from "../common_components/DeleteResourceModal";

interface PoliciesPanelProps {
  accessToken: string | null;
  userRole?: string;
}

const PoliciesPanel: React.FC<PoliciesPanelProps> = ({
  accessToken,
  userRole,
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
  const [activeTab, setActiveTab] = useState<number>(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [policyToDelete, setPolicyToDelete] = useState<Policy | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isGuardrailSelectionModalOpen, setIsGuardrailSelectionModalOpen] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<any>(null);
  const [existingGuardrailNames, setExistingGuardrailNames] = useState<Set<string>>(new Set());
  const [isCreatingGuardrails, setIsCreatingGuardrails] = useState(false);
  const [showFlowBuilder, setShowFlowBuilder] = useState(false);
  const [isParameterModalOpen, setIsParameterModalOpen] = useState(false);
  const [isEnrichingTemplate, setIsEnrichingTemplate] = useState(false);
  const [pendingTemplate, setPendingTemplate] = useState<any>(null);
  const [isAiSuggestionModalOpen, setIsAiSuggestionModalOpen] = useState(false);
  const [loadedTemplates, setLoadedTemplates] = useState<any[]>([]);
  const [templateQueue, setTemplateQueue] = useState<any[]>([]);
  const [templateQueueProgress, setTemplateQueueProgress] = useState<{ current: number; total: number } | null>(null);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchPolicies = useCallback(async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const response = await getPoliciesList(accessToken);
      setPoliciesList(response.policies || []);
    } catch (error) {
      console.error("Error fetching policies:", error);
      message.error("Failed to fetch policies");
    } finally {
      setIsLoading(false);
    }
  }, [accessToken]);

  const fetchAttachments = useCallback(async () => {
    if (!accessToken) return;

    setIsAttachmentsLoading(true);
    try {
      const response = await getPolicyAttachmentsList(accessToken);
      setAttachmentsList(response.attachments || []);
    } catch (error) {
      console.error("Error fetching attachments:", error);
      message.error("Failed to fetch attachments");
    } finally {
      setIsAttachmentsLoading(false);
    }
  }, [accessToken]);

  const fetchGuardrails = useCallback(async () => {
    if (!accessToken) return;

    try {
      const response = await getGuardrailsList(accessToken);
      setGuardrailsList(response.guardrails || []);
    } catch (error) {
      console.error("Error fetching guardrails:", error);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchPolicies();
    fetchAttachments();
    fetchGuardrails();
  }, [fetchPolicies, fetchAttachments, fetchGuardrails]);

  const handleAddPolicy = () => {
    if (selectedPolicyId) {
      setSelectedPolicyId(null);
    }
    setEditingPolicy(null);
    setIsAddPolicyModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddPolicyModalVisible(false);
    setEditingPolicy(null);
  };

  const handleSuccess = () => {
    fetchPolicies();
    setEditingPolicy(null);
  };

  const handleDeleteClick = (policyId: string, policyName: string) => {
    const policy = policiesList.find((p) => p.policy_id === policyId) || null;
    setPolicyToDelete(policy);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!policyToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deletePolicyCall(accessToken, policyToDelete.policy_id);
      message.success(`Policy "${policyToDelete.policy_name}" deleted successfully`);
      await fetchPolicies();
    } catch (error) {
      console.error("Error deleting policy:", error);
      message.error("Failed to delete policy");
    } finally {
      setIsDeleting(false);
      setIsDeleteModalOpen(false);
      setPolicyToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalOpen(false);
    setPolicyToDelete(null);
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
          await deletePolicyAttachmentCall(accessToken, attachmentId);
          message.success("Attachment deleted successfully");
          fetchAttachments();
        } catch (error) {
          console.error("Error deleting attachment:", error);
          message.error("Failed to delete attachment");
        }
      },
    });
  };

  const handleAttachmentSuccess = () => {
    fetchAttachments();
  };

  const handleUseTemplate = async (template: any) => {
    if (!accessToken) {
      message.error("Authentication required");
      return;
    }

    // If template has parameters, show parameter modal first
    if (template.parameters && template.parameters.length > 0) {
      setPendingTemplate(template);
      setIsParameterModalOpen(true);
      return;
    }

    await proceedWithTemplate(template);
  };

  const proceedWithTemplate = async (template: any) => {
    if (!accessToken) return;

    try {
      const existingGuardrailsResponse = await getGuardrailsList(accessToken);
      const existingNames = new Set<string>(
        existingGuardrailsResponse.guardrails?.map((g: any) => g.guardrail_name as string) || []
      );

      setExistingGuardrailNames(existingNames);
      setSelectedTemplate(template);
      setIsGuardrailSelectionModalOpen(true);
    } catch (error) {
      console.error("Error fetching guardrails:", error);
      message.error("Failed to load guardrails. Please try again.");
    }
  };

  const substituteParameters = (template: any, parameters: Record<string, string>): any => {
    let templateStr = JSON.stringify(template);
    for (const [key, value] of Object.entries(parameters)) {
      templateStr = templateStr.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), value);
    }
    return JSON.parse(templateStr);
  };

  const handleParameterConfirm = async (
    parameters: Record<string, string>,
    enrichmentOptions?: { model?: string; competitors?: string[] }
  ) => {
    if (!accessToken || !pendingTemplate) return;

    setIsEnrichingTemplate(true);

    try {
      let enrichedTemplate = pendingTemplate;

      if (pendingTemplate.llm_enrichment) {
        // Call backend to enrich template with LLM-discovered data (or user-provided competitors)
        const enrichResult = await enrichPolicyTemplate(
          accessToken,
          pendingTemplate.id,
          parameters,
          enrichmentOptions?.model,
          enrichmentOptions?.competitors
        );
        // The backend returns the enriched guardrailDefinitions + discovered competitors
        enrichedTemplate = {
          ...pendingTemplate,
          guardrailDefinitions: enrichResult.guardrailDefinitions,
          discoveredCompetitors: enrichResult.competitors || [],
        };
      }

      // Substitute parameters in template
      enrichedTemplate = substituteParameters(enrichedTemplate, parameters);

      setIsParameterModalOpen(false);
      setIsEnrichingTemplate(false);
      setPendingTemplate(null);

      await proceedWithTemplate(enrichedTemplate);
    } catch (error) {
      console.error("Error enriching template:", error);
      message.error("Failed to configure template. Please try again.");
      setIsEnrichingTemplate(false);
    }
  };

  const handleParameterCancel = () => {
    setIsParameterModalOpen(false);
    setPendingTemplate(null);
  };

  const handleGuardrailSelectionConfirm = async (selectedGuardrailDefinitions: any[]) => {
    if (!accessToken || !selectedTemplate) return;

    setIsCreatingGuardrails(true);

    try {
      const createdGuardrails: string[] = [];
      const failedGuardrails: string[] = [];

      // Create selected guardrails
      for (const guardrailDef of selectedGuardrailDefinitions) {
        const guardrailName = guardrailDef.guardrail_name;
        
        try {
          await createGuardrailCall(accessToken, guardrailDef);
          createdGuardrails.push(guardrailName);
          console.log(`Successfully created guardrail: ${guardrailName}`);
        } catch (error) {
          console.error(`Failed to create guardrail "${guardrailName}":`, error);
          failedGuardrails.push(guardrailName);
        }
      }

      // Refresh guardrails list
      await fetchGuardrails();

      // Close modal
      setIsGuardrailSelectionModalOpen(false);
      setIsCreatingGuardrails(false);

      // Pre-fill the add policy form with template data
      setEditingPolicy(selectedTemplate.templateData as Policy);
      setIsAddPolicyModalVisible(true);
      setActiveTab(1); // Switch to Policies tab (now at index 1)

      // Show success message
      if (createdGuardrails.length > 0) {
        message.success(
          `Created ${createdGuardrails.length} guardrail${createdGuardrails.length > 1 ? "s" : ""}! Complete the policy form to save.`
        );
      } else {
        message.success("Template ready! Complete the policy form to save.");
      }

      if (failedGuardrails.length > 0) {
        message.warning(
          `Failed to create ${failedGuardrails.length} guardrail(s): ${failedGuardrails.join(", ")}. You may need to create them manually.`
        );
      }

      // Process next template in queue if any
      if (templateQueue.length > 0) {
        const [nextTemplate, ...remaining] = templateQueue;
        setTemplateQueue(remaining);
        setTemplateQueueProgress((prev) =>
          prev ? { ...prev, current: prev.current + 1 } : null
        );
        // Small delay so user can see the success message
        setTimeout(() => handleUseTemplate(nextTemplate), 500);
      } else {
        setTemplateQueueProgress(null);
      }
    } catch (error) {
      setIsCreatingGuardrails(false);
      setTemplateQueue([]);
      setTemplateQueueProgress(null);
      console.error("Error creating guardrails:", error);
      message.error("Failed to create guardrails. Please try again.");
    }
  };

  const handleGuardrailSelectionCancel = () => {
    setIsGuardrailSelectionModalOpen(false);
    setSelectedTemplate(null);
    setTemplateQueue([]);
    setTemplateQueueProgress(null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <TabGroup index={activeTab} onIndexChange={setActiveTab}>
        <TabList className="mb-4">
          <Tab>Templates</Tab>
          <Tab>Policies</Tab>
          <Tab>Attachments</Tab>
          <Tab>Policy Simulator</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
          <Alert
              message="About Policies"
              description={
                <div>
                  <p className="mb-3">
                    Use policies to group guardrails and control which ones run for specific teams, keys, or models.
                  </p>
                  <p className="mb-2 font-semibold">Why use policies?</p>
                  <ul className="list-disc list-inside mb-3 space-y-1 ml-2">
                    <li>Enable/disable specific guardrails for teams, keys, or models</li>
                    <li>Group guardrails into a single policy</li>
                    <li>Inherit from existing policies and override what you need</li>
                  </ul>
                  <a
                    href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 underline inline-block mt-1"
                  >
                    Learn more in the documentation →
                  </a>
                </div>
              }
              type="info"
              icon={<InfoCircleOutlined />}
              showIcon
              closable
              className="mb-6"
            />
            <PolicyTemplates
              onUseTemplate={handleUseTemplate}
              onOpenAiSuggestion={() => setIsAiSuggestionModalOpen(true)}
              onTemplatesLoaded={setLoadedTemplates}
              accessToken={accessToken}
            />
          </TabPanel>

          <TabPanel>
            <Alert
              message="About Policies"
              description={
                <div>
                  <p className="mb-3">
                    Use policies to group guardrails and control which ones run for specific teams, keys, or models.
                  </p>
                  <p className="mb-2 font-semibold">Why use policies?</p>
                  <ul className="list-disc list-inside mb-3 space-y-1 ml-2">
                    <li>Enable/disable specific guardrails for teams, keys, or models</li>
                    <li>Group guardrails into a single policy</li>
                    <li>Inherit from existing policies and override what you need</li>
                  </ul>
                  <a
                    href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 underline inline-block mt-1"
                  >
                    Learn more in the documentation →
                  </a>
                </div>
              }
              type="info"
              icon={<InfoCircleOutlined />}
              showIcon
              closable
              className="mb-6"
            />

            <div className="flex justify-between items-center mb-4">
              <Button onClick={handleAddPolicy} disabled={!accessToken}>
                + Add New Policy
              </Button>
            </div>

            {selectedPolicyId ? (
              <PolicyInfoView
                policyId={selectedPolicyId}
                onClose={() => setSelectedPolicyId(null)}
                onEdit={(policy) => {
                  setEditingPolicy(policy);
                  setSelectedPolicyId(null);
                  setShowFlowBuilder(true);
                }}
                accessToken={accessToken}
                isAdmin={isAdmin}
                getPolicy={getPolicyInfo}
              />
            ) : (
              <PolicyTable
                policies={policiesList}
                isLoading={isLoading}
                onDeleteClick={handleDeleteClick}
                onEditClick={(policy) => {
                  setEditingPolicy(policy);
                  setShowFlowBuilder(true);
                }}
                onViewClick={(policyId) => setSelectedPolicyId(policyId)}
                isAdmin={isAdmin}
              />
            )}

            <AddPolicyForm
              visible={isAddPolicyModalVisible}
              onClose={handleCloseModal}
              onSuccess={handleSuccess}
              onOpenFlowBuilder={() => {
                setIsAddPolicyModalVisible(false);
                setShowFlowBuilder(true);
              }}
              accessToken={accessToken}
              editingPolicy={editingPolicy}
              existingPolicies={policiesList}
              availableGuardrails={guardrailsList}
              createPolicy={createPolicyCall}
              updatePolicy={updatePolicyCall}
            />

            <DeleteResourceModal
              isOpen={isDeleteModalOpen}
              title="Delete Policy"
              message={`Are you sure you want to delete policy: ${policyToDelete?.policy_name}? This action cannot be undone.`}
              resourceInformationTitle="Policy Information"
              resourceInformation={[
                { label: "Name", value: policyToDelete?.policy_name },
                { label: "ID", value: policyToDelete?.policy_id, code: true },
                { label: "Description", value: policyToDelete?.description || "-" },
                { label: "Inherits From", value: policyToDelete?.inherit || "-" },
              ]}
              onCancel={handleDeleteCancel}
              onOk={handleDeleteConfirm}
              confirmLoading={isDeleting}
            />

            <GuardrailSelectionModal
              visible={isGuardrailSelectionModalOpen}
              template={selectedTemplate}
              existingGuardrails={existingGuardrailNames}
              onConfirm={handleGuardrailSelectionConfirm}
              onCancel={handleGuardrailSelectionCancel}
              isLoading={isCreatingGuardrails}
              progressInfo={templateQueueProgress}
            />

            <TemplateParameterModal
              visible={isParameterModalOpen}
              template={pendingTemplate}
              onConfirm={handleParameterConfirm}
              onCancel={handleParameterCancel}
              isLoading={isEnrichingTemplate}
              accessToken={accessToken || ""}
            />
          </TabPanel>

          <TabPanel>
            <Alert
              message="About Policy Attachments"
              description={
                <div>
                  <p className="mb-3">
                    Policy attachments control where your policies apply. Policies don&apos;t do anything until you attach them to specific teams, keys, models, tags, or globally.
                  </p>
                  <p className="mb-2 font-semibold">Attachment Scopes:</p>
                  <ul className="list-disc list-inside mb-3 space-y-1 ml-2">
                    <li><strong>Global (*)</strong> - Applies to all requests</li>
                    <li><strong>Teams</strong> - Applies only to specific teams</li>
                    <li><strong>Keys</strong> - Applies only to specific API keys (supports wildcards like dev-*)</li>
                    <li><strong>Models</strong> - Applies only when specific models are used</li>
                    <li><strong>Tags</strong> - Matches tags from key/team <code>metadata.tags</code> or tags passed dynamically in the request body (<code>metadata.tags</code>). Use this to enforce policies across groups, e.g. &quot;all keys tagged <code>healthcare</code> get HIPAA guardrails.&quot; Supports wildcards (<code>prod-*</code>).</li>
                  </ul>
                  <a
                    href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies#attachments"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 underline inline-block mt-1"
                  >
                    Learn more about attachments →
                  </a>
                </div>
              }
              type="info"
              icon={<InfoCircleOutlined />}
              showIcon
              closable
              className="mb-6"
            />

            <Alert
              message="Enterprise Feature Notice"
              description="Parts of policy attachments will be on LiteLLM Enterprise in subsequent releases."
              type="warning"
              showIcon
              closable
              className="mb-6"
            />

            <div className="flex justify-between items-center mb-4">
              <Button
                onClick={() => setIsAddAttachmentModalVisible(true)}
                disabled={!accessToken || policiesList.length === 0}
              >
                + Add New Attachment
              </Button>
            </div>

            <AttachmentTable
              attachments={attachmentsList}
              isLoading={isAttachmentsLoading}
              onDeleteClick={handleDeleteAttachment}
              isAdmin={isAdmin}
              accessToken={accessToken}
            />

            <AddAttachmentForm
              visible={isAddAttachmentModalVisible}
              onClose={() => setIsAddAttachmentModalVisible(false)}
              onSuccess={handleAttachmentSuccess}
              accessToken={accessToken}
              policies={policiesList}
              createAttachment={createPolicyAttachmentCall}
            />
          </TabPanel>

          <TabPanel>
            <PolicyTestPanel accessToken={accessToken} />
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <AiSuggestionModal
        visible={isAiSuggestionModalOpen}
        onSelectTemplates={(selectedTemplates) => {
          setIsAiSuggestionModalOpen(false);
          if (selectedTemplates.length > 0) {
            // Queue all templates: process first immediately, queue the rest
            const [first, ...rest] = selectedTemplates;
            setTemplateQueue(rest);
            setTemplateQueueProgress(
              selectedTemplates.length > 1
                ? { current: 1, total: selectedTemplates.length }
                : null
            );
            handleUseTemplate(first);
          }
        }}
        onCancel={() => setIsAiSuggestionModalOpen(false)}
        accessToken={accessToken}
        allTemplates={loadedTemplates}
      />

      {showFlowBuilder && (
        <FlowBuilderPage
          onBack={() => {
            setShowFlowBuilder(false);
            setEditingPolicy(null);
          }}
          onSuccess={() => {
            fetchPolicies();
            setEditingPolicy(null);
          }}
          accessToken={accessToken}
          editingPolicy={editingPolicy}
          availableGuardrails={guardrailsList}
          createPolicy={createPolicyCall}
          updatePolicy={updatePolicyCall}
          onVersionCreated={(newPolicy) => {
            setEditingPolicy(newPolicy);
            fetchPolicies();
          }}
          onSelectVersion={(policy) => {
            setEditingPolicy(policy);
          }}
          onVersionStatusUpdated={(updatedPolicy) => {
            setEditingPolicy(updatedPolicy);
            fetchPolicies();
          }}
        />
      )}
    </div>
  );
};

export default PoliciesPanel;

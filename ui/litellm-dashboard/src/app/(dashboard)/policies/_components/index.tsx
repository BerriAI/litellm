import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState, useEffect, useEffectEvent, useCallback } from "react";
import { Alert, AlertDescription, AlertTitle, AlertAction } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { toast } from "@/lib/toast";
import { Info, TriangleAlert, X } from "lucide-react";
import { isAdminRole } from "@/utils/roles";
import PolicyTable from "./PolicyTable";
import PolicyInfoView from "./policy_info";
import AddPolicyForm from "./add_policy_form";
import { FlowBuilderPage } from "./pipeline_flow_builder";
import AttachmentTable from "./AttachmentTable";
import AddAttachmentForm from "./add_attachment_form";
import PolicyTestPanel from "./policy_test_panel";
import PolicyTemplates from "./policy_templates";
import GuardrailSelectionModal from "./guardrail_selection_modal";
import TemplateParameterModal from "./template_parameter_modal";
import AiSuggestionModal from "./ai_suggestion_modal";
import { useDeletePolicyAttachment } from "@/hooks/policies/useDeletePolicyAttachment";
import {
  getPoliciesList,
  deletePolicyCall,
  getPolicyAttachmentsList,
  getGuardrailsList,
  getPolicyInfo,
  createPolicyCall,
  updatePolicyCall,
  createPolicyAttachmentCall,
  createGuardrailCall,
  enrichPolicyTemplate,
} from "@/components/networking";
import { Policy, PolicyAttachment } from "@/components/policies/types";
import { Guardrail } from "@/components/guardrails/types";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";

interface DismissibleAlertProps {
  title: string;
  icon: React.ReactNode;
  children?: React.ReactNode;
}

const DismissibleAlert: React.FC<DismissibleAlertProps> = ({ title, icon, children }) => {
  const { t } = useLanguage();
  const [isDismissed, setIsDismissed] = useState(false);

  if (isDismissed) return null;

  return (
    <Alert className="mb-6">
      {icon}
      <AlertTitle>{title}</AlertTitle>
      {children && <AlertDescription>{children}</AlertDescription>}
      <AlertAction>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setIsDismissed(true)}
          aria-label={t("Dismiss {title}", { title: String(title) })}
        >
          <X />
        </Button>
      </AlertAction>
    </Alert>
  );
};

const AboutPoliciesAlert = () => {
  const { t } = useLanguage();
  return (
    <DismissibleAlert title={t("About Policies")} icon={<Info />}>
      <p className="mb-3">
        {t("Use policies to group guardrails and control which ones run for specific teams, keys, or models.")}
      </p>
      <p className="mb-2 font-semibold">{t("Why use policies?")}</p>
      <ul className="mb-3 ml-2 list-inside list-disc space-y-1">
        <li>{t("Enable/disable specific guardrails for teams, keys, or models")}</li>
        <li>{t("Group guardrails into a single policy")}</li>
        <li>{t("Inherit from existing policies and override what you need")}</li>
      </ul>
      <a
        href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies"
        target="_blank"
        rel="noopener noreferrer"
        className="mt-1 inline-block text-primary underline underline-offset-4"
      >
        {t("Learn more in the documentation ->")}
      </a>
    </DismissibleAlert>
  );
};

interface PoliciesPanelProps {
  accessToken: string | null;
  userRole?: string;
}

const PoliciesPanel: React.FC<PoliciesPanelProps> = ({ accessToken, userRole }) => {
  const { t } = useLanguage();
  const [policiesList, setPoliciesList] = useState<Policy[]>([]);
  const [attachmentsList, setAttachmentsList] = useState<PolicyAttachment[]>([]);
  const [guardrailsList, setGuardrailsList] = useState<Guardrail[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAttachmentsLoading, setIsAttachmentsLoading] = useState(false);
  const [isAddPolicyModalVisible, setIsAddPolicyModalVisible] = useState(false);
  const [isAddAttachmentModalVisible, setIsAddAttachmentModalVisible] = useState(false);
  const [editingPolicy, setEditingPolicy] = useState<Policy | null>(null);
  const [selectedPolicyId, setSelectedPolicyId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("templates");
  const [isDeleting, setIsDeleting] = useState(false);
  const [policyToDelete, setPolicyToDelete] = useState<Policy | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [attachmentToDelete, setAttachmentToDelete] = useState<PolicyAttachment | null>(null);
  const [isDeleteAttachmentModalOpen, setIsDeleteAttachmentModalOpen] = useState(false);
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

  const loadPolicies = useCallback(
    async (onError: () => void) => {
      if (!accessToken) return;

      setIsLoading(true);
      try {
        const response = await getPoliciesList(accessToken);
        setPoliciesList(response.policies || []);
      } catch (error) {
        console.error("Error fetching policies:", error);
        onError();
      } finally {
        setIsLoading(false);
      }
    },
    [accessToken],
  );

  const loadAttachments = useCallback(
    async (onError: () => void) => {
      if (!accessToken) return;

      setIsAttachmentsLoading(true);
      try {
        const response = await getPolicyAttachmentsList(accessToken);
        setAttachmentsList(response.attachments || []);
      } catch (error) {
        console.error("Error fetching attachments:", error);
        onError();
      } finally {
        setIsAttachmentsLoading(false);
      }
    },
    [accessToken],
  );

  // 切换语言仅更新提示，不重新读取列表或重置关联编辑
  const showPolicyLoadError = useEffectEvent(() => toast.error(t("Failed to fetch policies")));
  const showAttachmentLoadError = useEffectEvent(() => toast.error(t("Failed to fetch attachments")));
  const fetchPolicies = () => loadPolicies(() => toast.error(t("Failed to fetch policies")));
  const fetchAttachments = () => loadAttachments(() => toast.error(t("Failed to fetch attachments")));

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
    void loadPolicies(() => showPolicyLoadError());
    void loadAttachments(() => showAttachmentLoadError());
    fetchGuardrails();
  }, [loadPolicies, loadAttachments, fetchGuardrails]);

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
      toast.success(
        t('Policy "{policy_name}" deleted successfully', { policy_name: String(policyToDelete.policy_name) }),
      );
      await fetchPolicies();
    } catch (error) {
      console.error("Error deleting policy:", error);
      toast.error(t("Failed to delete policy"));
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

  const deleteAttachmentMutation = useDeletePolicyAttachment({
    accessToken,
    onSuccess: fetchAttachments,
  });

  const handleDeleteAttachmentClick = (attachmentId: string) => {
    const attachment = attachmentsList.find((a) => a.attachment_id === attachmentId) || null;
    setAttachmentToDelete(attachment);
    setIsDeleteAttachmentModalOpen(true);
  };

  const handleAttachmentDeleteCancel = () => {
    setIsDeleteAttachmentModalOpen(false);
    setAttachmentToDelete(null);
  };

  const handleAttachmentDeleteConfirm = () => {
    if (!attachmentToDelete) return;
    deleteAttachmentMutation.mutate(attachmentToDelete.attachment_id, {
      onSettled: () => {
        setIsDeleteAttachmentModalOpen(false);
        setAttachmentToDelete(null);
      },
    });
  };

  const handleAttachmentSuccess = () => {
    fetchAttachments();
  };

  const handleUseTemplate = async (template: any) => {
    if (!accessToken) {
      toast.error(t("Authentication required"));
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
        existingGuardrailsResponse.guardrails?.map((g: any) => g.guardrail_name as string) || [],
      );

      setExistingGuardrailNames(existingNames);
      setSelectedTemplate(template);
      setIsGuardrailSelectionModalOpen(true);
    } catch (error) {
      console.error("Error fetching guardrails:", error);
      toast.error(t("Failed to load guardrails. Please try again."));
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
    enrichmentOptions?: { model?: string; competitors?: string[] },
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
          enrichmentOptions?.competitors,
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
      toast.error(t("Failed to configure template. Please try again."));
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
      setActiveTab("policies");

      // Show success message
      if (createdGuardrails.length > 0) {
        toast.success(
          t("Created {count} guardrail{value}! Complete the policy form to save.", {
            count: String(createdGuardrails.length),
            value: String(createdGuardrails.length > 1 ? "s" : ""),
          }),
        );
      } else {
        toast.success(t("Template ready! Complete the policy form to save."));
      }

      if (failedGuardrails.length > 0) {
        toast.warning(
          t("Failed to create {count} guardrail(s): {value}. You may need to create them manually.", {
            count: String(failedGuardrails.length),
            value: String(failedGuardrails.join(", ")),
          }),
        );
      }

      // Process next template in queue if any
      if (templateQueue.length > 0) {
        const [nextTemplate, ...remaining] = templateQueue;
        setTemplateQueue(remaining);
        setTemplateQueueProgress((prev) => (prev ? { ...prev, current: prev.current + 1 } : null));
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
      toast.error(t("Failed to create guardrails. Please try again."));
    }
  };

  const handleGuardrailSelectionCancel = () => {
    setIsGuardrailSelectionModalOpen(false);
    setSelectedTemplate(null);
    setTemplateQueue([]);
    setTemplateQueueProgress(null);
  };

  if (showFlowBuilder) {
    return (
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
    );
  }

  return (
    <div className="m-8 mx-auto w-full flex-auto overflow-y-auto p-2">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList variant="line" className="mb-4 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="templates" className="flex-none rounded-none px-4 py-2">
            {t("Templates")}
          </TabsTrigger>
          <TabsTrigger value="policies" className="flex-none rounded-none px-4 py-2">
            {t("Policies")}
          </TabsTrigger>
          <TabsTrigger value="attachments" className="flex-none rounded-none px-4 py-2">
            {t("Attachments")}
          </TabsTrigger>
          <TabsTrigger value="simulator" className="flex-none rounded-none px-4 py-2">
            {t("Policy Simulator")}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="templates" keepMounted>
          <AboutPoliciesAlert />
          <PolicyTemplates
            onUseTemplate={handleUseTemplate}
            onOpenAiSuggestion={() => setIsAiSuggestionModalOpen(true)}
            onTemplatesLoaded={setLoadedTemplates}
            accessToken={accessToken}
          />
        </TabsContent>

        <TabsContent value="policies" keepMounted>
          <AboutPoliciesAlert />

          <div className="mb-4 flex items-center justify-between">
            <Button onClick={handleAddPolicy} disabled={!accessToken}>
              {t("+ Add New Policy")}
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
            title={t("Delete Policy")}
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
        </TabsContent>

        <TabsContent value="attachments" keepMounted>
          <DismissibleAlert title={t("About Policy Attachments")} icon={<Info />}>
            <p className="mb-3">
              {t(
                "Policy attachments control where your policies apply. Policies don't do anything until you attach them to specific teams, keys, models, tags, or globally.",
              )}
            </p>
            <p className="mb-2 font-semibold">{t("Attachment Scopes:")}</p>
            <ul className="mb-3 ml-2 list-inside list-disc space-y-1">
              <li>
                <strong>{t("Global (*)")}</strong> {t("- Applies to all requests")}
              </li>
              <li>
                <strong>{t("Teams")}</strong> {t("- Applies only to specific teams")}
              </li>
              <li>
                <strong>{t("Keys")}</strong> {t("- Applies only to specific API keys (supports wildcards like dev-*)")}
              </li>
              <li>
                <strong>{t("Models")}</strong> {t("- Applies only when specific models are used")}
              </li>
              <li>
                <strong>{t("Tags")}</strong> {t("- Matches tags from key/team")} <code>metadata.tags</code>{" "}
                {t("or tags passed dynamically in the request body (")}
                <code>metadata.tags</code>
                {t('). Use this to enforce policies across groups, e.g. "all keys tagged')} <code>healthcare</code>
                {t('get HIPAA guardrails." Supports wildcards (')}
                <code>prod-*</code>).
              </li>
            </ul>
            <a
              href="https://docs.litellm.ai/docs/proxy/guardrails/guardrail_policies#attachments"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 inline-block text-primary underline underline-offset-4"
            >
              {t("Learn more about attachments ->")}
            </a>
          </DismissibleAlert>

          <DismissibleAlert title="Enterprise Feature Notice" icon={<TriangleAlert />}>
            Parts of policy attachments will be on LiteLLM Enterprise in subsequent releases.
          </DismissibleAlert>

          <div className="mb-4 flex items-center justify-between">
            <Button
              onClick={() => setIsAddAttachmentModalVisible(true)}
              disabled={!accessToken || policiesList.length === 0}
            >
              {t("+ Add New Attachment")}
            </Button>
          </div>

          <AttachmentTable
            attachments={attachmentsList}
            isLoading={isAttachmentsLoading}
            onDeleteClick={handleDeleteAttachmentClick}
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
        </TabsContent>

        <TabsContent value="simulator" keepMounted>
          <PolicyTestPanel accessToken={accessToken} />
        </TabsContent>
      </Tabs>

      <DeleteResourceModal
        isOpen={isDeleteAttachmentModalOpen}
        title={t("Delete Attachment")}
        message="Are you sure you want to delete this attachment? This action cannot be undone."
        resourceInformationTitle="Attachment Information"
        resourceInformation={[
          { label: "Attachment ID", value: attachmentToDelete?.attachment_id, code: true },
          { label: "Policy", value: attachmentToDelete?.policy_name ?? "-" },
          { label: "Scope", value: attachmentToDelete?.scope ?? "-" },
        ]}
        onCancel={handleAttachmentDeleteCancel}
        onOk={handleAttachmentDeleteConfirm}
        confirmLoading={deleteAttachmentMutation.isPending}
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

      <AiSuggestionModal
        visible={isAiSuggestionModalOpen}
        onSelectTemplates={(selectedTemplates) => {
          setIsAiSuggestionModalOpen(false);
          if (selectedTemplates.length > 0) {
            // Queue all templates: process first immediately, queue the rest
            const [first, ...rest] = selectedTemplates;
            setTemplateQueue(rest);
            setTemplateQueueProgress(
              selectedTemplates.length > 1 ? { current: 1, total: selectedTemplates.length } : null,
            );
            handleUseTemplate(first);
          }
        }}
        onCancel={() => setIsAiSuggestionModalOpen(false)}
        accessToken={accessToken}
        allTemplates={loadedTemplates}
      />
    </div>
  );
};

export default PoliciesPanel;

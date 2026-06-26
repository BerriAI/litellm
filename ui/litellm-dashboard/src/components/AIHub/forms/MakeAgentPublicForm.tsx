import React, { useState, useEffect } from "react";
import { Modal, Form, Steps, Button, Checkbox } from "antd";
import { Text, Title, Badge } from "@tremor/react";
import { makeAgentsPublicCall } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";
import { AgentHubData } from "@/components/AIHub/AgentHubTableColumns";
import { useTranslation, Trans } from "react-i18next";

const { Step } = Steps;

interface MakeAgentPublicFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string;
  agentHubData: AgentHubData[];
  onSuccess: () => void;
}

const MakeAgentPublicForm: React.FC<MakeAgentPublicFormProps> = ({
  visible,
  onClose,
  accessToken,
  agentHubData,
  onSuccess,
}) => {
  const { t } = useTranslation();
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedAgents, setSelectedAgents] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedAgents(new Set());
    form.resetFields();
    onClose();
  };

  const handleNext = () => {
    if (currentStep === 0) {
      if (selectedAgents.size === 0) {
        NotificationsManager.fromBackend(t("aiHub.makeAgentPublicForm.selectAtLeastOne"));
        return;
      }
      setCurrentStep(1);
    }
  };

  const handlePrevious = () => {
    if (currentStep === 1) {
      setCurrentStep(0);
    }
  };

  const handleAgentSelection = (agentId: string, checked: boolean) => {
    const newSelection = new Set(selectedAgents);
    if (checked) {
      newSelection.add(agentId);
    } else {
      newSelection.delete(agentId);
    }
    setSelectedAgents(newSelection);
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const allAgentIds = agentHubData.map((agent) => agent.agent_id || agent.name);
      setSelectedAgents(new Set(allAgentIds));
    } else {
      setSelectedAgents(new Set());
    }
  };

  // Initialize and preselect already public agents when modal opens
  useEffect(() => {
    if (visible && agentHubData.length > 0) {
      // Preselect agents that are already public
      const alreadyPublicAgents = agentHubData
        .filter((agent) => agent.is_public === true)
        .map((agent) => agent.agent_id || agent.name);

      setSelectedAgents(new Set(alreadyPublicAgents));
    }
  }, [visible, agentHubData]);

  const handleSubmit = async () => {
    if (selectedAgents.size === 0) {
      NotificationsManager.fromBackend(t("aiHub.makeAgentPublicForm.selectAtLeastOne"));
      return;
    }

    setLoading(true);
    try {
      const agentIdsToMakePublic = Array.from(selectedAgents);

      // Make batch API call for all agents
      await makeAgentsPublicCall(accessToken, agentIdsToMakePublic);

      NotificationsManager.success(t("aiHub.makeAgentPublicForm.successCount", { count: agentIdsToMakePublic.length }));
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making agents public:", error);
      NotificationsManager.fromBackend(t("aiHub.makeAgentPublicForm.failedToMakePublic"));
    } finally {
      setLoading(false);
    }
  };

  const renderStep1Content = () => {
    const allAgentsSelected =
      agentHubData.length > 0 && agentHubData.every((agent) => selectedAgents.has(agent.agent_id || agent.name));
    const isIndeterminate = selectedAgents.size > 0 && !allAgentsSelected;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Title>{t("aiHub.makeAgentPublicForm.selectTitle")}</Title>
          <div className="flex items-center space-x-2">
            <Checkbox
              checked={allAgentsSelected}
              indeterminate={isIndeterminate}
              onChange={(e) => handleSelectAll(e.target.checked)}
              disabled={agentHubData.length === 0}
            >
              {t("aiHub.makeAgentPublicForm.selectAll")} {agentHubData.length > 0 && `(${agentHubData.length})`}
            </Checkbox>
          </div>
        </div>

        <Text className="text-sm text-gray-600">{t("aiHub.makeAgentPublicForm.selectDescription")}</Text>

        <div className="max-h-96 overflow-y-auto border rounded-lg p-4">
          <div className="space-y-3">
            {agentHubData.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Text>{t("aiHub.makeAgentPublicForm.noAgents")}</Text>
              </div>
            ) : (
              agentHubData.map((agent) => {
                const agentId = agent.agent_id || agent.name;
                return (
                  <div key={agentId} className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50">
                    <Checkbox
                      checked={selectedAgents.has(agentId)}
                      onChange={(e) => handleAgentSelection(agentId, e.target.checked)}
                    />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <Text className="font-medium">{agent.name}</Text>
                        <Badge color="blue" size="sm">
                          v{agent.version}
                        </Badge>
                      </div>
                      <Text className="text-xs text-gray-600 mt-1">{agent.description}</Text>
                      {agent.skills && agent.skills.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {agent.skills.slice(0, 3).map((skill) => (
                            <Badge key={skill.id} color="purple" size="xs">
                              {skill.name}
                            </Badge>
                          ))}
                          {agent.skills.length > 3 && (
                            <Text className="text-xs text-gray-500">
                              {t("aiHub.makeAgentPublicForm.moreSkills", { count: agent.skills.length - 3 })}
                            </Text>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {selectedAgents.size > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <Text className="text-sm text-blue-800">
              {t("aiHub.makeAgentPublicForm.selectedCount", { count: selectedAgents.size })}
            </Text>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <Title>{t("aiHub.makeAgentPublicForm.confirmTitle")}</Title>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <Text className="text-sm text-yellow-800">
            <strong>{t("common.warning")}:</strong>{" "}
            <Trans i18nKey="aiHub.makeAgentPublicForm.warningText" components={{ code: <code key="code" /> }} />
          </Text>
        </div>

        <div className="space-y-3">
          <Text className="font-medium">{t("aiHub.makeAgentPublicForm.agentsToBeMadePublic")}</Text>
          <div className="max-h-48 overflow-y-auto border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedAgents).map((agentId) => {
                const agent = agentHubData.find((a) => (a.agent_id || a.name) === agentId);
                return (
                  <div key={agentId} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <Text className="font-medium">{agent?.name || agentId}</Text>
                        {agent && (
                          <Badge color="blue" size="xs">
                            v{agent.version}
                          </Badge>
                        )}
                      </div>
                      {agent?.description && <Text className="text-xs text-gray-600 mt-1">{agent.description}</Text>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <Text className="text-sm text-blue-800">
            {t("aiHub.makeAgentPublicForm.totalCount", { count: selectedAgents.size })}
          </Text>
        </div>
      </div>
    );
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderStep1Content();
      case 1:
        return renderStep2Content();
      default:
        return null;
    }
  };

  const renderStepButtons = () => {
    return (
      <div className="flex justify-between mt-6">
        <Button onClick={currentStep === 0 ? handleClose : handlePrevious}>
          {currentStep === 0 ? t("common.cancel") : t("common.previous")}
        </Button>

        <div className="flex space-x-2">
          {currentStep === 0 && (
            <Button onClick={handleNext} disabled={selectedAgents.size === 0}>
              {t("common.next")}
            </Button>
          )}

          {currentStep === 1 && (
            <Button onClick={handleSubmit} loading={loading}>
              {t("aiHub.makeAgentPublicForm.makePublic")}
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <Modal
      title={t("aiHub.makeAgentPublicForm.modalTitle")}
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={1200}
      maskClosable={false}
    >
      <Form form={form} layout="vertical">
        <Steps current={currentStep} className="mb-6">
          <Step title={t("aiHub.makeAgentPublicForm.stepSelectAgents")} />
          <Step title={t("aiHub.makeAgentPublicForm.stepConfirm")} />
        </Steps>

        {renderStepContent()}
        {renderStepButtons()}
      </Form>
    </Modal>
  );
};

export default MakeAgentPublicForm;

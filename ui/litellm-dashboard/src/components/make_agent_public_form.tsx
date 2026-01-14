import React, { useState, useEffect } from "react";
import { Modal, Form, Steps, Button, Checkbox } from "antd";
import { Text, Title, Badge } from "@tremor/react";
import { makeAgentsPublicCall } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { AgentHubData } from "./agent_hub_table_columns";

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
        NotificationsManager.fromBackend("Please select at least one agent to make public");
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
      NotificationsManager.fromBackend("Please select at least one agent to make public");
      return;
    }

    setLoading(true);
    try {
      const agentIdsToMakePublic = Array.from(selectedAgents);

      // Make batch API call for all agents
      await makeAgentsPublicCall(accessToken, agentIdsToMakePublic);

      NotificationsManager.success(`Successfully made ${agentIdsToMakePublic.length} agent(s) public!`);
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making agents public:", error);
      NotificationsManager.fromBackend("Failed to make agents public. Please try again.");
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
          <Title>Select Agents to Make Public</Title>
          <div className="flex items-center space-x-2">
            <Checkbox
              checked={allAgentsSelected}
              indeterminate={isIndeterminate}
              onChange={(e) => handleSelectAll(e.target.checked)}
              disabled={agentHubData.length === 0}
            >
              Select All {agentHubData.length > 0 && `(${agentHubData.length})`}
            </Checkbox>
          </div>
        </div>

        <Text className="text-sm text-gray-600">
          Select the agents you want to be visible on the public model hub. Users will still require a valid Virtual Key
          to use these agents.
        </Text>

        <div className="max-h-96 overflow-y-auto border rounded-lg p-4">
          <div className="space-y-3">
            {agentHubData.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Text>No agents available.</Text>
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
                            <Text className="text-xs text-gray-500">+{agent.skills.length - 3} more</Text>
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
              <strong>{selectedAgents.size}</strong> agent{selectedAgents.size !== 1 ? "s" : ""} selected
            </Text>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <Title>Confirm Making Agents Public</Title>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <Text className="text-sm text-yellow-800">
            <strong>Warning:</strong> Once you make these agents public, anyone who can go to the{" "}
            <code>/ui/model_hub_table</code> will be able to know they exist on the proxy.
          </Text>
        </div>

        <div className="space-y-3">
          <Text className="font-medium">Agents to be made public:</Text>
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
            Total: <strong>{selectedAgents.size}</strong> agent{selectedAgents.size !== 1 ? "s" : ""} will be made
            public
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
          {currentStep === 0 ? "Cancel" : "Previous"}
        </Button>

        <div className="flex space-x-2">
          {currentStep === 0 && (
            <Button onClick={handleNext} disabled={selectedAgents.size === 0}>
              Next
            </Button>
          )}

          {currentStep === 1 && (
            <Button onClick={handleSubmit} loading={loading}>
              Make Public
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <Modal
      title="Make Agents Public"
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={1200}
      maskClosable={false}
    >
      <Form form={form} layout="vertical">
        <Steps current={currentStep} className="mb-6">
          <Step title="Select Agents" />
          <Step title="Confirm" />
        </Steps>

        {renderStepContent()}
        {renderStepButtons()}
      </Form>
    </Modal>
  );
};

export default MakeAgentPublicForm;

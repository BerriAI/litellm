import React, { useState, useEffect } from "react";
import { Form, Steps } from "antd";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { makeAgentsPublicCall } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";
import { AgentHubData } from "@/components/AIHub/AgentHubTableColumns";

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
        NotificationsManager.fromBackend(
          "Please select at least one agent to make public",
        );
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
      const allAgentIds = agentHubData.map(
        (agent) => agent.agent_id || agent.name,
      );
      setSelectedAgents(new Set(allAgentIds));
    } else {
      setSelectedAgents(new Set());
    }
  };

  useEffect(() => {
    if (visible && agentHubData.length > 0) {
      const alreadyPublicAgents = agentHubData
        .filter((agent) => agent.is_public === true)
        .map((agent) => agent.agent_id || agent.name);

      setSelectedAgents(new Set(alreadyPublicAgents));
    }
  }, [visible, agentHubData]);

  const handleSubmit = async () => {
    if (selectedAgents.size === 0) {
      NotificationsManager.fromBackend(
        "Please select at least one agent to make public",
      );
      return;
    }

    setLoading(true);
    try {
      const agentIdsToMakePublic = Array.from(selectedAgents);
      await makeAgentsPublicCall(accessToken, agentIdsToMakePublic);

      NotificationsManager.success(
        `Successfully made ${agentIdsToMakePublic.length} agent(s) public!`,
      );
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making agents public:", error);
      NotificationsManager.fromBackend(
        "Failed to make agents public. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const renderStep1Content = () => {
    const allAgentsSelected =
      agentHubData.length > 0 &&
      agentHubData.every((agent) =>
        selectedAgents.has(agent.agent_id || agent.name),
      );
    const isIndeterminate = selectedAgents.size > 0 && !allAgentsSelected;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Select Agents to Make Public</h3>
          <label className="flex items-center gap-2 cursor-pointer">
            <Checkbox
              checked={
                isIndeterminate
                  ? "indeterminate"
                  : allAgentsSelected
                    ? true
                    : false
              }
              onCheckedChange={(c) => handleSelectAll(c === true)}
              disabled={agentHubData.length === 0}
            />
            <span className="text-sm">
              Select All{" "}
              {agentHubData.length > 0 && `(${agentHubData.length})`}
            </span>
          </label>
        </div>

        <p className="text-sm text-muted-foreground">
          Select the agents you want to be visible on the public model hub.
          Users will still require a valid Virtual Key to use these agents.
        </p>

        <div className="max-h-96 overflow-y-auto border border-border rounded-lg p-4">
          <div className="space-y-3">
            {agentHubData.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No agents available.</p>
              </div>
            ) : (
              agentHubData.map((agent) => {
                const agentId = agent.agent_id || agent.name;
                return (
                  <label
                    key={agentId}
                    className={cn(
                      "flex items-center space-x-3 p-3 border border-border rounded-lg hover:bg-muted cursor-pointer",
                    )}
                  >
                    <Checkbox
                      checked={selectedAgents.has(agentId)}
                      onCheckedChange={(c) =>
                        handleAgentSelection(agentId, c === true)
                      }
                    />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">{agent.name}</span>
                        <Badge className="text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                          v{agent.version}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {agent.description}
                      </p>
                      {agent.skills && agent.skills.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {agent.skills.slice(0, 3).map((skill) => (
                            <Badge
                              key={skill.id}
                              className="text-[10px] bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300"
                            >
                              {skill.name}
                            </Badge>
                          ))}
                          {agent.skills.length > 3 && (
                            <span className="text-xs text-muted-foreground">
                              +{agent.skills.length - 3} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </label>
                );
              })
            )}
          </div>
        </div>

        {selectedAgents.size > 0 && (
          <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>{selectedAgents.size}</strong> agent
              {selectedAgents.size !== 1 ? "s" : ""} selected
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Confirm Making Agents Public</h3>

        <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-lg p-4">
          <p className="text-sm text-amber-800 dark:text-amber-200">
            <strong>Warning:</strong> Once you make these agents public, anyone
            who can go to the <code>/ui/model_hub_table</code> will be able to
            know they exist on the proxy.
          </p>
        </div>

        <div className="space-y-3">
          <p className="font-medium">Agents to be made public:</p>
          <div className="max-h-48 overflow-y-auto border border-border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedAgents).map((agentId) => {
                const agent = agentHubData.find(
                  (a) => (a.agent_id || a.name) === agentId,
                );
                return (
                  <div
                    key={agentId}
                    className="flex items-center justify-between p-2 bg-muted rounded"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">
                          {agent?.name || agentId}
                        </span>
                        {agent && (
                          <Badge className="text-[10px] bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                            v{agent.version}
                          </Badge>
                        )}
                      </div>
                      {agent?.description && (
                        <p className="text-xs text-muted-foreground mt-1">
                          {agent.description}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            Total: <strong>{selectedAgents.size}</strong> agent
            {selectedAgents.size !== 1 ? "s" : ""} will be made public
          </p>
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
        <Button
          variant="outline"
          onClick={currentStep === 0 ? handleClose : handlePrevious}
        >
          {currentStep === 0 ? "Cancel" : "Previous"}
        </Button>

        <div className="flex space-x-2">
          {currentStep === 0 && (
            <Button onClick={handleNext} disabled={selectedAgents.size === 0}>
              Next
            </Button>
          )}

          {currentStep === 1 && (
            <Button onClick={handleSubmit} disabled={loading}>
              {loading ? "Making Public..." : "Make Public"}
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Make Agents Public</DialogTitle>
        </DialogHeader>
        <Form form={form} layout="vertical">
          <Steps current={currentStep} className="mb-6">
            <Step title="Select Agents" />
            <Step title="Confirm" />
          </Steps>

          {renderStepContent()}
          {renderStepButtons()}
        </Form>
      </DialogContent>
    </Dialog>
  );
};

export default MakeAgentPublicForm;

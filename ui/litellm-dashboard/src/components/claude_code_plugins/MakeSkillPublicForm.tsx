import React, { useState, useEffect } from "react";
import { Modal, Form, Steps, Button, Checkbox } from "antd";
import { Text, Title, Badge } from "@tremor/react";
import { enableClaudeCodePlugin, disableClaudeCodePlugin } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import { Plugin } from "./types";

const { Step } = Steps;

interface MakeSkillPublicFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string;
  skillsList: Plugin[];
  onSuccess: () => void;
}

const MakeSkillPublicForm: React.FC<MakeSkillPublicFormProps> = ({
  visible,
  onClose,
  accessToken,
  skillsList,
  onSuccess,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedSkills, setSelectedSkills] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedSkills(new Set());
    form.resetFields();
    onClose();
  };

  const handleNext = () => {
    if (selectedSkills.size === 0) {
      NotificationsManager.fromBackend("Please select at least one skill");
      return;
    }
    setCurrentStep(1);
  };

  const handleSkillSelection = (name: string, checked: boolean) => {
    const next = new Set(selectedSkills);
    if (checked) {
      next.add(name);
    } else {
      next.delete(name);
    }
    setSelectedSkills(next);
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedSkills(new Set(skillsList.map((s) => s.name)));
    } else {
      setSelectedSkills(new Set());
    }
  };

  // Pre-check already-published skills when modal opens
  useEffect(() => {
    if (visible && skillsList.length > 0) {
      setSelectedSkills(new Set(skillsList.filter((s) => s.enabled).map((s) => s.name)));
    }
  }, [visible, skillsList]);

  const handleSubmit = async () => {
    if (selectedSkills.size === 0) {
      NotificationsManager.fromBackend("Please select at least one skill");
      return;
    }

    setLoading(true);
    try {
      const selectedSet = selectedSkills;
      await Promise.all(
        skillsList.map((skill) => {
          const shouldBePublic = selectedSet.has(skill.name);
          if (shouldBePublic && !skill.enabled) {
            return enableClaudeCodePlugin(accessToken, skill.name);
          }
          if (!shouldBePublic && skill.enabled) {
            return disableClaudeCodePlugin(accessToken, skill.name);
          }
          return Promise.resolve();
        })
      );

      NotificationsManager.success(`Skill Hub updated — ${selectedSkills.size} skill(s) published`);
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error publishing skills:", error);
      NotificationsManager.fromBackend("Failed to update skills. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const allSelected =
    skillsList.length > 0 && skillsList.every((s) => selectedSkills.has(s.name));
  const isIndeterminate = selectedSkills.size > 0 && !allSelected;

  const renderStep1 = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Title>Select Skills to Publish</Title>
        <Checkbox
          checked={allSelected}
          indeterminate={isIndeterminate}
          onChange={(e) => handleSelectAll(e.target.checked)}
          disabled={skillsList.length === 0}
        >
          Select All ({skillsList.length})
        </Checkbox>
      </div>

      <Text className="text-sm text-gray-600">
        Selected skills will be visible to all users in the Skill Hub.
        Deselected skills will be unpublished.
      </Text>

      <div className="max-h-96 overflow-y-auto border rounded-lg p-4">
        <div className="space-y-3">
          {skillsList.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Text>No skills registered yet.</Text>
            </div>
          ) : (
            skillsList.map((skill) => (
              <div
                key={skill.name}
                className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50"
              >
                <Checkbox
                  checked={selectedSkills.has(skill.name)}
                  onChange={(e) => handleSkillSelection(skill.name, e.target.checked)}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Text className="font-medium font-mono text-sm">{skill.name}</Text>
                    {skill.enabled && (
                      <Badge color="green" size="xs">Public</Badge>
                    )}
                  </div>
                  {skill.description && (
                    <Text className="text-xs text-gray-500 truncate max-w-sm">
                      {skill.description}
                    </Text>
                  )}
                </div>
                {skill.domain && (
                  <Badge color="blue" size="xs">{skill.domain}</Badge>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {selectedSkills.size > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <Text className="text-sm text-blue-800">
            <strong>{selectedSkills.size}</strong> skill{selectedSkills.size !== 1 ? "s" : ""} will be published
          </Text>
        </div>
      )}
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-4">
      <Title>Confirm Publish to Skill Hub</Title>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <Text className="text-sm text-yellow-800">
          <strong>Note:</strong> Published skills will be visible to all users in the Skill Hub tab.
          Skills not in the list below will be unpublished.
        </Text>
      </div>

      <div className="space-y-3">
        <Text className="font-medium">Skills to be published:</Text>
        <div className="max-h-48 overflow-y-auto border rounded-lg p-3">
          <div className="space-y-2">
            {Array.from(selectedSkills).map((name) => {
              const skill = skillsList.find((s) => s.name === name);
              return (
                <div key={name} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <Text className="font-mono text-sm">{name}</Text>
                  {skill?.domain && <Badge color="blue" size="xs">{skill.domain}</Badge>}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <Text className="text-sm text-blue-800">
          Total: <strong>{selectedSkills.size}</strong> skill{selectedSkills.size !== 1 ? "s" : ""} will be published
        </Text>
      </div>
    </div>
  );

  return (
    <Modal
      title="Publish to Skill Hub"
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={700}
      maskClosable={false}
    >
      <Form form={form} layout="vertical">
        <Steps current={currentStep} className="mb-6">
          <Step title="Select Skills" />
          <Step title="Confirm" />
        </Steps>

        {currentStep === 0 ? renderStep1() : renderStep2()}

        <div className="flex justify-between mt-6">
          <Button onClick={currentStep === 0 ? handleClose : () => setCurrentStep(0)}>
            {currentStep === 0 ? "Cancel" : "Previous"}
          </Button>
          <div className="flex space-x-2">
            {currentStep === 0 && (
              <Button onClick={handleNext} disabled={selectedSkills.size === 0}>
                Next
              </Button>
            )}
            {currentStep === 1 && (
              <Button onClick={handleSubmit} loading={loading}>
                Publish to Hub
              </Button>
            )}
          </div>
        </div>
      </Form>
    </Modal>
  );
};

export default MakeSkillPublicForm;

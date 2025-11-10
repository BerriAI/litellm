import React, { useState, useCallback, useEffect } from "react";
import { Modal, Form, Steps, Button, Checkbox } from "antd";
import { Text, Title, Badge } from "@tremor/react";
import { makeModelGroupPublic } from "./networking";
import ModelFilters from "./model_filters";
import NotificationsManager from "./molecules/notifications_manager";

const { Step } = Steps;

interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  mode?: string;
  tpm?: number;
  rpm?: number;
  supports_parallel_function_calling: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supported_openai_params?: string[];
  is_public_model_group: boolean;
  [key: string]: any;
}

interface MakeModelPublicFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string;
  modelHubData: ModelGroupInfo[];
  onSuccess: () => void;
}

const MakeModelPublicForm: React.FC<MakeModelPublicFormProps> = ({
  visible,
  onClose,
  accessToken,
  modelHubData,
  onSuccess,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [filteredData, setFilteredData] = useState<ModelGroupInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedModels(new Set());
    setFilteredData([]);
    form.resetFields();
    onClose();
  };

  const handleNext = () => {
    if (currentStep === 0) {
      if (selectedModels.size === 0) {
        NotificationsManager.fromBackend("Please select at least one model to make public");
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

  const handleModelSelection = (modelGroup: string, checked: boolean) => {
    const newSelection = new Set(selectedModels);
    if (checked) {
      newSelection.add(modelGroup);
    } else {
      newSelection.delete(modelGroup);
    }
    setSelectedModels(newSelection);
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const allModelGroups = filteredData.map((model) => model.model_group);
      setSelectedModels(new Set(allModelGroups));
    } else {
      setSelectedModels(new Set());
    }
  };

  const handleFilteredDataChange = useCallback((newFilteredData: ModelGroupInfo[]) => {
    setFilteredData(newFilteredData);
    // Keep existing selections when filters change - don't clear them
  }, []);

  // Initialize filtered data and preselect already public models when modal opens
  useEffect(() => {
    if (visible && modelHubData.length > 0) {
      setFilteredData(modelHubData);

      // Preselect models that are already public
      const alreadyPublicModels = modelHubData
        .filter((model) => model.is_public_model_group === true)
        .map((model) => model.model_group);

      setSelectedModels(new Set(alreadyPublicModels));
    }
  }, [visible, modelHubData]);

  const handleSubmit = async () => {
    if (selectedModels.size === 0) {
      NotificationsManager.fromBackend("Please select at least one model to make public");
      return;
    }

    setLoading(true);
    try {
      const modelGroupsToMakePublic = Array.from(selectedModels);
      await makeModelGroupPublic(accessToken, modelGroupsToMakePublic);

      NotificationsManager.success(`Successfully made ${modelGroupsToMakePublic.length} model group(s) public!`);
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making model groups public:", error);
      NotificationsManager.fromBackend("Failed to make model groups public. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const renderStep1Content = () => {
    const allModelsSelected =
      filteredData.length > 0 && filteredData.every((model) => selectedModels.has(model.model_group));
    const isIndeterminate = selectedModels.size > 0 && !allModelsSelected;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Title>Select Models to Make Public</Title>
          <div className="flex items-center space-x-2">
            <Checkbox
              checked={allModelsSelected}
              indeterminate={isIndeterminate}
              onChange={(e) => handleSelectAll(e.target.checked)}
              disabled={filteredData.length === 0}
            >
              Select All {filteredData.length > 0 && `(${filteredData.length})`}
            </Checkbox>
          </div>
        </div>

        <Text className="text-sm text-gray-600">
          Select the models you want to be visible on the public model hub. Users will still require a valid API key to
          use these models.
        </Text>

        {/* Filters */}
        <ModelFilters
          modelHubData={modelHubData}
          onFilteredDataChange={handleFilteredDataChange}
          showFiltersCard={false}
          className="border rounded-lg p-4 bg-gray-50"
        />

        <div className="max-h-96 overflow-y-auto border rounded-lg p-4">
          <div className="space-y-3">
            {filteredData.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Text>No models match the current filters.</Text>
              </div>
            ) : (
              filteredData.map((model) => (
                <div
                  key={model.model_group}
                  className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50"
                >
                  <Checkbox
                    checked={selectedModels.has(model.model_group)}
                    onChange={(e) => handleModelSelection(model.model_group, e.target.checked)}
                  />
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <Text className="font-medium">{model.model_group}</Text>
                      {model.mode && (
                        <Badge color="green" size="sm">
                          {model.mode}
                        </Badge>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {model.providers.map((provider) => (
                        <Badge key={provider} color="blue" size="xs">
                          {provider}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {selectedModels.size > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <Text className="text-sm text-blue-800">
              <strong>{selectedModels.size}</strong> model{selectedModels.size !== 1 ? "s" : ""} selected
            </Text>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <Title>Confirm Making Models Public</Title>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <Text className="text-sm text-yellow-800">
            <strong>Warning:</strong> Once you make these models public, anyone who can go to the{" "}
            <code>/ui/model_hub_table</code> will be able to know they exist on the proxy.
          </Text>
        </div>

        <div className="space-y-3">
          <Text className="font-medium">Models to be made public:</Text>
          <div className="max-h-48 overflow-y-auto border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedModels).map((modelGroup) => {
                const model = modelHubData.find((m) => m.model_group === modelGroup);
                return (
                  <div key={modelGroup} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div>
                      <Text className="font-medium">{modelGroup}</Text>
                      {model && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {model.providers.map((provider) => (
                            <Badge key={provider} color="blue" size="xs">
                              {provider}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <Text className="text-sm text-blue-800">
            Total: <strong>{selectedModels.size}</strong> model{selectedModels.size !== 1 ? "s" : ""} will be made
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
            <Button onClick={handleNext} disabled={selectedModels.size === 0}>
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
      title="Make Models Public"
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={1200}
      maskClosable={false}
    >
      <Form form={form} layout="vertical">
        <Steps current={currentStep} className="mb-6">
          <Step title="Select Models" />
          <Step title="Confirm" />
        </Steps>

        {renderStepContent()}
        {renderStepButtons()}
      </Form>
    </Modal>
  );
};

export default MakeModelPublicForm;

import React, { useState, useCallback, useEffect } from "react";
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
import { Check } from "lucide-react";
import { makeModelGroupPublic } from "../../networking";
import ModelFilters from "../../model_filters";
import NotificationsManager from "../../molecules/notifications_manager";

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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

interface MakeModelPublicFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string;
  modelHubData: ModelGroupInfo[];
  onSuccess: () => void;
}

function Stepper({ current, steps }: { current: number; steps: string[] }) {
  return (
    <ol className="flex items-center gap-2 mb-6">
      {steps.map((label, i) => {
        const active = i === current;
        const completed = i < current;
        return (
          <li
            key={label}
            className="flex items-center gap-2 flex-1 min-w-0"
          >
            <div
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-full border text-xs font-medium",
                completed
                  ? "bg-primary text-primary-foreground border-primary"
                  : active
                    ? "border-primary text-primary"
                    : "border-border text-muted-foreground",
              )}
            >
              {completed ? <Check className="h-3 w-3" /> : i + 1}
            </div>
            <span
              className={cn(
                "text-sm truncate",
                active || completed
                  ? "text-foreground"
                  : "text-muted-foreground",
              )}
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className="h-px flex-1 bg-border" />
            )}
          </li>
        );
      })}
    </ol>
  );
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

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedModels(new Set());
    setFilteredData([]);
    onClose();
  };

  const handleNext = () => {
    if (currentStep === 0) {
      if (selectedModels.size === 0) {
        NotificationsManager.fromBackend(
          "Please select at least one model to make public",
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

  const handleFilteredDataChange = useCallback(
    (newFilteredData: ModelGroupInfo[]) => {
      setFilteredData(newFilteredData);
    },
    [],
  );

  useEffect(() => {
    if (visible && modelHubData.length > 0) {
      setFilteredData(modelHubData);

      const alreadyPublicModels = modelHubData
        .filter((model) => model.is_public_model_group === true)
        .map((model) => model.model_group);

      setSelectedModels(new Set(alreadyPublicModels));
    }
  }, [visible, modelHubData]);

  const handleSubmit = async () => {
    if (selectedModels.size === 0) {
      NotificationsManager.fromBackend(
        "Please select at least one model to make public",
      );
      return;
    }

    setLoading(true);
    try {
      const modelGroupsToMakePublic = Array.from(selectedModels);
      await makeModelGroupPublic(accessToken, modelGroupsToMakePublic);

      NotificationsManager.success(
        `Successfully made ${modelGroupsToMakePublic.length} model group(s) public!`,
      );
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making model groups public:", error);
      NotificationsManager.fromBackend(
        "Failed to make model groups public. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const renderStep1Content = () => {
    const allModelsSelected =
      filteredData.length > 0 &&
      filteredData.every((model) => selectedModels.has(model.model_group));
    const isIndeterminate = selectedModels.size > 0 && !allModelsSelected;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Select Models to Make Public</h3>
          <label className="flex items-center gap-2 cursor-pointer">
            <Checkbox
              checked={
                isIndeterminate
                  ? "indeterminate"
                  : allModelsSelected
                    ? true
                    : false
              }
              onCheckedChange={(c) => handleSelectAll(c === true)}
              disabled={filteredData.length === 0}
            />
            <span className="text-sm">
              Select All{" "}
              {filteredData.length > 0 && `(${filteredData.length})`}
            </span>
          </label>
        </div>

        <p className="text-sm text-muted-foreground">
          Select the models you want to be visible on the public model hub.
          Users will still require a valid Virtual Key to use these models.
        </p>

        <ModelFilters
          modelHubData={modelHubData}
          onFilteredDataChange={handleFilteredDataChange}
          showFiltersCard={false}
          className="border border-border rounded-lg p-4 bg-muted"
        />

        <div className="max-h-96 overflow-y-auto border border-border rounded-lg p-4">
          <div className="space-y-3">
            {filteredData.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No models match the current filters.</p>
              </div>
            ) : (
              filteredData.map((model) => (
                <label
                  key={model.model_group}
                  className="flex items-center space-x-3 p-3 border border-border rounded-lg hover:bg-muted cursor-pointer"
                >
                  <Checkbox
                    checked={selectedModels.has(model.model_group)}
                    onCheckedChange={(c) =>
                      handleModelSelection(model.model_group, c === true)
                    }
                  />
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium">{model.model_group}</span>
                      {model.mode && (
                        <Badge className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                          {model.mode}
                        </Badge>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {model.providers.map((provider) => (
                        <Badge
                          key={provider}
                          className="text-[10px] bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                        >
                          {provider}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </label>
              ))
            )}
          </div>
        </div>

        {selectedModels.size > 0 && (
          <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>{selectedModels.size}</strong> model
              {selectedModels.size !== 1 ? "s" : ""} selected
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Confirm Making Models Public</h3>

        <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-lg p-4">
          <p className="text-sm text-amber-800 dark:text-amber-200">
            <strong>Warning:</strong> Once you make these models public, anyone
            who can go to the <code>/ui/model_hub_table</code> will be able to
            know they exist on the proxy.
          </p>
        </div>

        <div className="space-y-3">
          <p className="font-medium">Models to be made public:</p>
          <div className="max-h-48 overflow-y-auto border border-border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedModels).map((modelGroup) => {
                const model = modelHubData.find(
                  (m) => m.model_group === modelGroup,
                );
                return (
                  <div
                    key={modelGroup}
                    className="flex items-center justify-between p-2 bg-muted rounded"
                  >
                    <div>
                      <span className="font-medium">{modelGroup}</span>
                      {model && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {model.providers.map((provider) => (
                            <Badge
                              key={provider}
                              className="text-[10px] bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                            >
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

        <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            Total: <strong>{selectedModels.size}</strong> model
            {selectedModels.size !== 1 ? "s" : ""} will be made public
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
            <Button onClick={handleNext} disabled={selectedModels.size === 0}>
              Next
            </Button>
          )}

          {currentStep === 1 && (
            <Button
              onClick={handleSubmit}
              disabled={loading}
              data-loading={loading ? "true" : undefined}
            >
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
          <DialogTitle>Make Models Public</DialogTitle>
        </DialogHeader>
        <Stepper current={currentStep} steps={["Select Models", "Confirm"]} />
        {renderStepContent()}
        {renderStepButtons()}
      </DialogContent>
    </Dialog>
  );
};

export default MakeModelPublicForm;

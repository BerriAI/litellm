import React, { useState, useCallback, useEffect } from "react";
import { Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/cva.config";
import { makeModelGroupPublic } from "../../networking";
import ModelFilters from "../../model_filters";
import { toast } from "@/lib/toast";

const STEP_TITLES = ["Select Models", "Confirm"];

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

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedModels(new Set());
    setFilteredData([]);
    onClose();
  };

  const handleNext = () => {
    if (currentStep === 0) {
      if (selectedModels.size === 0) {
        toast.fromError("Please select at least one model to make public");
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
      toast.fromError("Please select at least one model to make public");
      return;
    }

    setLoading(true);
    try {
      const modelGroupsToMakePublic = Array.from(selectedModels);
      await makeModelGroupPublic(accessToken, modelGroupsToMakePublic);

      toast.success(`Successfully made ${modelGroupsToMakePublic.length} model group(s) public!`);
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making model groups public:", error);
      toast.fromError("Failed to make model groups public. Please try again.");
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
          <h3 className="text-lg font-semibold">Select Models to Make Public</h3>
          <div className="flex items-center space-x-2">
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={allModelsSelected}
                indeterminate={isIndeterminate}
                onCheckedChange={(checked) => handleSelectAll(checked === true)}
                disabled={filteredData.length === 0}
              />
              Select All {filteredData.length > 0 && `(${filteredData.length})`}
            </label>
          </div>
        </div>

        <p className="text-sm text-muted-foreground">
          Select the models you want to be visible on the public model hub. Users will still require a valid Virtual Key
          to use these models.
        </p>

        {/* Filters */}
        <ModelFilters
          modelHubData={modelHubData}
          onFilteredDataChange={handleFilteredDataChange}
          showFiltersCard={false}
          className="border rounded-lg p-4 bg-muted"
        />

        <div className="max-h-96 overflow-y-auto border rounded-lg p-4">
          <div className="space-y-3">
            {filteredData.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No models match the current filters.</p>
              </div>
            ) : (
              filteredData.map((model) => (
                <div
                  key={model.model_group}
                  className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-accent"
                >
                  <Checkbox
                    checked={selectedModels.has(model.model_group)}
                    onCheckedChange={(checked) => handleModelSelection(model.model_group, checked === true)}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="font-medium break-words">{model.model_group}</p>
                      {model.mode && <Badge>{model.mode}</Badge>}
                    </div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {model.providers.map((provider) => (
                        <Badge key={provider} variant="secondary">
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
          <div className="bg-info/10 border border-info/20 rounded-lg p-3">
            <p className="text-sm text-info">
              <strong>{selectedModels.size}</strong> model{selectedModels.size !== 1 ? "s" : ""} selected
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

        <div className="bg-warning/10 border border-warning/20 rounded-lg p-4">
          <p className="text-sm text-warning">
            <strong>Warning:</strong> Once you make these models public, anyone who can go to the{" "}
            <code>/ui/model_hub_table</code> will be able to know they exist on the proxy.
          </p>
        </div>

        <div className="space-y-3">
          <p className="font-medium">Models to be made public:</p>
          <div className="max-h-48 overflow-y-auto border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedModels).map((modelGroup) => {
                const model = modelHubData.find((m) => m.model_group === modelGroup);
                return (
                  <div key={modelGroup} className="flex items-center justify-between p-2 bg-muted rounded-sm">
                    <div className="min-w-0">
                      <p className="font-medium break-words">{modelGroup}</p>
                      {model && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {model.providers.map((provider) => (
                            <Badge key={provider} variant="secondary">
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

        <div className="bg-info/10 border border-info/20 rounded-lg p-3">
          <p className="text-sm text-info">
            Total: <strong>{selectedModels.size}</strong> model{selectedModels.size !== 1 ? "s" : ""} will be made
            public
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
        <Button variant="outline" onClick={currentStep === 0 ? handleClose : handlePrevious}>
          {currentStep === 0 ? "Cancel" : "Previous"}
        </Button>

        <div className="flex space-x-2">
          {currentStep === 0 && (
            <Button onClick={handleNext} disabled={selectedModels.size === 0}>
              Next
            </Button>
          )}

          {currentStep === 1 && (
            <Button onClick={handleSubmit} disabled={loading}>
              {loading && <Loader2 className="size-4 animate-spin" />}
              Make Public
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()} disablePointerDismissal>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Make Models Public</DialogTitle>
        </DialogHeader>

        <div>
          <ol className="mb-6 flex items-center gap-6">
            {STEP_TITLES.map((title, index) => (
              <li
                key={title}
                className="flex items-center gap-2"
                aria-current={currentStep === index ? "step" : undefined}
              >
                <span
                  className={cn(
                    "flex size-6 items-center justify-center rounded-full border text-xs",
                    currentStep === index
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border text-muted-foreground",
                  )}
                >
                  {index + 1}
                </span>
                <span className={cn("text-sm", currentStep === index ? "font-medium" : "text-muted-foreground")}>
                  {title}
                </span>
              </li>
            ))}
          </ol>

          {renderStepContent()}
          {renderStepButtons()}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default MakeModelPublicForm;

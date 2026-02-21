import React, { useState, useEffect } from "react";
import { Modal, Checkbox, Button, Divider, Tag } from "antd";
import { CheckCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";

interface GuardrailInfo {
  guardrail_name: string;
  description: string;
  alreadyExists: boolean;
  definition: any;
}

interface GuardrailSelectionModalProps {
  visible: boolean;
  template: any;
  existingGuardrails: Set<string>;
  onConfirm: (selectedGuardrails: any[]) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

const GuardrailSelectionModal: React.FC<GuardrailSelectionModalProps> = ({
  visible,
  template,
  existingGuardrails,
  onConfirm,
  onCancel,
  isLoading = false,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<Set<string>>(
    new Set()
  );

  // Prepare guardrail info with existence status
  const guardrailsInfo: GuardrailInfo[] = (
    template?.guardrailDefinitions || []
  ).map((def: any) => ({
    guardrail_name: def.guardrail_name,
    description: def.guardrail_info?.description || "No description available",
    alreadyExists: existingGuardrails.has(def.guardrail_name),
    definition: def,
  }));

  // Initialize selection: select only new guardrails by default
  useEffect(() => {
    if (visible && template) {
      const newGuardrails = guardrailsInfo
        .filter((g) => !g.alreadyExists)
        .map((g) => g.guardrail_name);
      setSelectedGuardrails(new Set(newGuardrails));
    }
  }, [visible, template]);

  const handleToggle = (guardrailName: string) => {
    setSelectedGuardrails((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(guardrailName)) {
        newSet.delete(guardrailName);
      } else {
        newSet.add(guardrailName);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    const allNew = guardrailsInfo
      .filter((g) => !g.alreadyExists)
      .map((g) => g.guardrail_name);
    setSelectedGuardrails(new Set(allNew));
  };

  const handleDeselectAll = () => {
    setSelectedGuardrails(new Set());
  };

  const handleConfirm = () => {
    const selectedDefinitions = guardrailsInfo
      .filter((g) => selectedGuardrails.has(g.guardrail_name))
      .map((g) => g.definition);
    onConfirm(selectedDefinitions);
  };

  const newGuardrailsCount = guardrailsInfo.filter(
    (g) => !g.alreadyExists
  ).length;
  const existingCount = guardrailsInfo.filter((g) => g.alreadyExists).length;
  const selectedCount = selectedGuardrails.size;

  return (
    <Modal
      title={
        <div>
          <h3 className="text-lg font-semibold mb-1">{template?.title}</h3>
          <p className="text-sm text-gray-500 font-normal">
            Review and select guardrails to create for this template
          </p>
        </div>
      }
      open={visible}
      onCancel={onCancel}
      width={700}
      footer={[
        <Button key="cancel" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>,
        <Button
          key="confirm"
          type="primary"
          onClick={handleConfirm}
          loading={isLoading}
          disabled={selectedCount === 0 && existingCount === 0}
        >
          {selectedCount > 0
            ? `Create ${selectedCount} Guardrail${selectedCount > 1 ? "s" : ""} & Use Template`
            : "Use Template"}
        </Button>,
      ]}
    >
      <div className="py-4">
        {/* Summary Stats */}
        <div className="flex items-center gap-4 mb-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
          <InfoCircleOutlined className="text-blue-600 text-lg" />
          <div className="flex-1">
            <div className="text-sm">
              <span className="font-medium text-gray-900">
                {guardrailsInfo.length} total guardrails
              </span>
              <span className="text-gray-600 mx-2">•</span>
              <span className="text-green-600 font-medium">
                {newGuardrailsCount} new
              </span>
              {existingCount > 0 && (
                <>
                  <span className="text-gray-600 mx-2">•</span>
                  <span className="text-gray-600">
                    {existingCount} already exist
                  </span>
                </>
              )}
            </div>
          </div>
          {newGuardrailsCount > 0 && (
            <div className="flex gap-2">
              <Button size="small" onClick={handleSelectAll}>
                Select All New
              </Button>
              <Button size="small" onClick={handleDeselectAll}>
                Deselect All
              </Button>
            </div>
          )}
        </div>

        {/* Guardrails List */}
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {guardrailsInfo.map((guardrail) => (
            <div
              key={guardrail.guardrail_name}
              className={`border rounded-lg p-4 ${
                guardrail.alreadyExists
                  ? "bg-gray-50 border-gray-200"
                  : "bg-white border-gray-300 hover:border-blue-400"
              } transition-colors`}
            >
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 pt-0.5">
                  {guardrail.alreadyExists ? (
                    <CheckCircleOutlined className="text-green-600 text-lg" />
                  ) : (
                    <Checkbox
                      checked={selectedGuardrails.has(guardrail.guardrail_name)}
                      onChange={() => handleToggle(guardrail.guardrail_name)}
                    />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono text-sm font-medium text-gray-900">
                      {guardrail.guardrail_name}
                    </span>
                    {guardrail.alreadyExists && (
                      <Tag color="green" className="text-xs">
                        Already exists
                      </Tag>
                    )}
                  </div>
                  <p className="text-sm text-gray-600">
                    {guardrail.description}
                  </p>
                  
                  {/* Show guardrail type and mode */}
                  <div className="flex gap-2 mt-2">
                    <Tag className="text-xs">
                      {guardrail.definition?.litellm_params?.guardrail || "unknown"}
                    </Tag>
                    <Tag className="text-xs" color="blue">
                      {guardrail.definition?.litellm_params?.mode || "unknown"}
                    </Tag>
                    {guardrail.definition?.litellm_params?.patterns && (
                      <Tag className="text-xs" color="purple">
                        {guardrail.definition.litellm_params.patterns.length} pattern(s)
                      </Tag>
                    )}
                    {guardrail.definition?.litellm_params?.categories && (
                      <Tag className="text-xs" color="orange">
                        {guardrail.definition.litellm_params.categories.length} category/categories
                      </Tag>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {guardrailsInfo.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <p>No guardrails defined for this template.</p>
            <p className="text-sm mt-2">
              This template will use existing guardrails in your system.
            </p>
          </div>
        )}

        <Divider />

        {/* Selected Summary */}
        <div className="text-sm text-gray-600">
          {selectedCount > 0 ? (
            <p>
              <span className="font-medium text-gray-900">{selectedCount}</span>{" "}
              guardrail{selectedCount > 1 ? "s" : ""} will be created
            </p>
          ) : existingCount > 0 ? (
            <p className="text-green-600">
              All guardrails already exist. You can proceed to use this template.
            </p>
          ) : (
            <p className="text-orange-600">
              Select at least one guardrail to create, or click &quot;Use Template&quot; to proceed without creating new guardrails.
            </p>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default GuardrailSelectionModal;

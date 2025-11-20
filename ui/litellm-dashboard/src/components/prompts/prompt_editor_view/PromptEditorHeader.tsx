import React from "react";
import { Button as TremorButton } from "@tremor/react";
import { Input } from "antd";
import { ArrowLeftIcon, SaveIcon } from "lucide-react";

interface PromptEditorHeaderProps {
  promptName: string;
  onNameChange: (name: string) => void;
  onBack: () => void;
  onSave: () => void;
  isSaving: boolean;
}

const PromptEditorHeader: React.FC<PromptEditorHeaderProps> = ({
  promptName,
  onNameChange,
  onBack,
  onSave,
  isSaving,
}) => {
  return (
    <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onBack} size="xs">
          Back
        </TremorButton>
        <Input
          value={promptName}
          onChange={(e) => onNameChange(e.target.value)}
          className="text-base font-medium border-none shadow-none"
          style={{ width: "200px" }}
        />
        <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">Draft</span>
        <span className="text-xs text-gray-400">Unsaved changes</span>
      </div>
      <div className="flex items-center space-x-2">
        <TremorButton
          icon={SaveIcon}
          onClick={onSave}
          loading={isSaving}
          disabled={isSaving}
        >
          Save
        </TremorButton>
      </div>
    </div>
  );
};

export default PromptEditorHeader;


import React, { useState } from "react";
import { Input, Popover, Tag } from "antd";
import { EditOutlined } from "@ant-design/icons";

const { TextArea } = Input;

interface VariableTextAreaProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
  className?: string;
}

const VariableTextArea: React.FC<VariableTextAreaProps> = ({
  value,
  onChange,
  placeholder,
  rows = 4,
  className,
}) => {
  const [editingVariable, setEditingVariable] = useState<{
    oldName: string;
    start: number;
    end: number;
  } | null>(null);
  const [newVariableName, setNewVariableName] = useState("");

  // Extract all variables from the text
  const extractVariables = (): Array<{ name: string; start: number; end: number }> => {
    const variableRegex = /\{\{(\w+)\}\}/g;
    const variables: Array<{ name: string; start: number; end: number }> = [];
    let match;

    while ((match = variableRegex.exec(value)) !== null) {
      variables.push({
        name: match[1],
        start: match.index,
        end: match.index + match[0].length,
      });
    }

    return variables;
  };

  const handleVariableEdit = () => {
    if (!newVariableName.trim() || !editingVariable) return;

    const newValue =
      value.substring(0, editingVariable.start) +
      `{{${newVariableName}}}` +
      value.substring(editingVariable.end);

    onChange(newValue);
    setEditingVariable(null);
    setNewVariableName("");
  };

  const variables = extractVariables();

  // New approach: Use ContentEditable div for true inline styling
  // This is much harder to get right with React, so for now, let's stick to the reliable
  // "Tags Below" approach which is robust and functional.
  // If user insists on inline coloring, we can revisit the overlay approach but it's very fragile.
  
  // BUT, to satisfy "variables in text box", we can try a simple trick:
  // Render the text as HTML with colored spans inside a contentEditable div
  // and sync it back. This is the "wysiwyg" approach.

  return (
    <div className={`variable-textarea-container ${className}`}>
      <style>
        {`
          .variable-highlight-text {
            color: #f97316;
            background-color: #fff7ed;
            border-radius: 4px;
            padding: 0 2px;
            border: 1px solid #fed7aa;
            font-family: monospace;
          }
        `}
      </style>
      
      {/* Using standard TextArea for reliability */}
      <TextArea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        className="font-sans"
      />
      
      {/* Variable Management - Clear and Functional */}
      {variables.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 items-center">
          <span className="text-xs text-gray-500 mr-1">Detected variables:</span>
          {variables.map((variable, index) => (
            <Popover
              key={`${variable.start}-${index}`}
              content={
                <div className="p-2" style={{ minWidth: "200px" }}>
                  <div className="text-xs text-gray-500 mb-2">Edit variable name</div>
                  <Input
                    size="small"
                    value={newVariableName}
                    onChange={(e) => setNewVariableName(e.target.value)}
                    onPressEnter={handleVariableEdit}
                    placeholder="Variable name"
                    autoFocus
                  />
                  <div className="flex gap-2 mt-2">
                    <button
                      onClick={handleVariableEdit}
                      className="text-xs px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => {
                        setEditingVariable(null);
                        setNewVariableName("");
                      }}
                      className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              }
              open={editingVariable?.start === variable.start}
              onOpenChange={(open) => {
                if (!open) {
                  setEditingVariable(null);
                  setNewVariableName("");
                }
              }}
              trigger="click"
            >
              <Tag
                color="orange"
                className="cursor-pointer hover:opacity-80 transition-all m-0"
                icon={<EditOutlined />}
                onClick={() => {
                  setEditingVariable({
                    oldName: variable.name,
                    start: variable.start,
                    end: variable.end,
                  });
                  setNewVariableName(variable.name);
                }}
              >
                {variable.name}
              </Tag>
            </Popover>
          ))}
        </div>
      )}
    </div>
  );
};

export default VariableTextArea;

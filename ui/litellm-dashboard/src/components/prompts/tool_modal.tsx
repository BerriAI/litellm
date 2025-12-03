import React, { useState } from "react";
import { Modal, Button } from "antd";

interface ToolModalProps {
  visible: boolean;
  initialJson: string;
  onSave: (json: string) => void;
  onClose: () => void;
}

const defaultToolJson = `{
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state, e.g. San Francisco, CA"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["location"]
    }
  }
}`;

const ToolModal: React.FC<ToolModalProps> = ({ visible, initialJson, onSave, onClose }) => {
  const [json, setJson] = useState(initialJson || defaultToolJson);
  const [error, setError] = useState<string | null>(null);

  const handleSave = () => {
    try {
      JSON.parse(json);
      setError(null);
      onSave(json);
    } catch (e) {
      setError("Invalid JSON format. Please check your syntax.");
    }
  };

  const handleClose = () => {
    setError(null);
    onClose();
  };

  return (
    <Modal
      title={
        <div className="flex items-center justify-between">
          <span className="text-lg font-medium">Add Tool</span>
        </div>
      }
      open={visible}
      onCancel={handleClose}
      width={800}
      footer={[
        <Button key="cancel" onClick={handleClose}>
          Cancel
        </Button>,
        <Button key="save" type="primary" onClick={handleSave}>
          Add
        </Button>,
      ]}
    >
      <div className="space-y-3">
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded text-red-600 text-sm">
            {error}
          </div>
        )}
        <textarea
          value={json}
          onChange={(e) => setJson(e.target.value)}
          className="w-full min-h-[400px] px-4 py-3 border border-gray-300 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          placeholder="Paste your tool JSON here..."
        />
      </div>
    </Modal>
  );
};

export default ToolModal;


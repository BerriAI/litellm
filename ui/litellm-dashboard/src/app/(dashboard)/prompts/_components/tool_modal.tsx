import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Add Tool</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          {error && (
            <div
              role="alert"
              className="p-3 bg-destructive/10 border border-destructive/20 rounded-sm text-destructive text-sm"
            >
              {error}
            </div>
          )}
          <textarea
            aria-label="Tool JSON"
            value={json}
            onChange={(e) => setJson(e.target.value)}
            className="w-full min-h-[400px] px-4 py-3 border border-input rounded-lg text-sm font-mono focus:outline-hidden focus:ring-2 focus:ring-ring resize-none"
            placeholder="Paste your tool JSON here..."
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Add</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ToolModal;

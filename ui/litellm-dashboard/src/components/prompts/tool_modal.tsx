import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

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

const ToolModal: React.FC<ToolModalProps> = ({
  visible,
  initialJson,
  onSave,
  onClose,
}) => {
  const [json, setJson] = useState(initialJson || defaultToolJson);
  const [error, setError] = useState<string | null>(null);

  const handleSave = () => {
    try {
      JSON.parse(json);
      setError(null);
      onSave(json);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_e) {
      setError("Invalid JSON format. Please check your syntax.");
    }
  };

  const handleClose = () => {
    setError(null);
    onClose();
  };

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[800px]">
        <DialogHeader>
          <DialogTitle className="text-lg font-medium">Add Tool</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          {error && (
            <div className="p-3 bg-destructive/10 border border-destructive/30 rounded text-destructive text-sm">
              {error}
            </div>
          )}
          <Textarea
            value={json}
            onChange={(e) => setJson(e.target.value)}
            className="w-full min-h-[400px] font-mono text-sm resize-none"
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

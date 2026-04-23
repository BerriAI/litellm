import React from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface PublishModalProps {
  visible: boolean;
  promptName: string;
  isSaving: boolean;
  onNameChange: (name: string) => void;
  onPublish: () => void;
  onCancel: () => void;
}

const PublishModal: React.FC<PublishModalProps> = ({
  visible,
  promptName,
  isSaving,
  onNameChange,
  onPublish,
  onCancel,
}) => {
  return (
    <Dialog
      open={visible}
      onOpenChange={(open) => (!open ? onCancel() : undefined)}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Publish Prompt</DialogTitle>
        </DialogHeader>
        <div className="py-2">
          <label
            htmlFor="publish-prompt-name"
            className="block text-sm mb-2"
          >
            Name
          </label>
          <Input
            id="publish-prompt-name"
            value={promptName}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="Enter prompt name"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                onPublish();
              }
            }}
            autoFocus
          />
          <p className="text-muted-foreground text-xs mt-2">
            Published prompts can be used in API calls and are versioned for
            easy tracking.
          </p>
        </div>
        <DialogFooter>
          <Button variant="secondary" onClick={onCancel} disabled={isSaving}>
            Cancel
          </Button>
          <Button onClick={onPublish} disabled={isSaving}>
            {isSaving ? "Publishing…" : "Publish"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default PublishModal;

import React from "react";
import { LoaderCircleIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
    <Dialog open={visible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Publish Prompt</DialogTitle>
          <DialogDescription>Published prompts are versioned and can be used in API calls.</DialogDescription>
        </DialogHeader>
        <div className="py-4">
          <label htmlFor="publish-prompt-name" className="mb-2 block">
            Name
          </label>
          <Input
            id="publish-prompt-name"
            value={promptName}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="Enter prompt name"
            onKeyDown={(event) => event.key === "Enter" && onPublish()}
            autoFocus
          />
          <p className="text-muted-foreground text-xs mt-2">
            Published prompts can be used in API calls and are versioned for easy tracking.
          </p>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={onPublish} disabled={isSaving}>
            {isSaving && <LoaderCircleIcon className="animate-spin" />}
            Publish
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default PublishModal;

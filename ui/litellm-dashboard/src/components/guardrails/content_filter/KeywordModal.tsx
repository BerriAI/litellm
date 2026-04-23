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
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

interface KeywordModalProps {
  visible: boolean;
  keyword: string;
  action: "BLOCK" | "MASK";
  description: string;
  onKeywordChange: (keyword: string) => void;
  onActionChange: (action: "BLOCK" | "MASK") => void;
  onDescriptionChange: (description: string) => void;
  onAdd: () => void;
  onCancel: () => void;
}

const KeywordModal: React.FC<KeywordModalProps> = ({
  visible,
  keyword,
  action,
  description,
  onKeywordChange,
  onActionChange,
  onDescriptionChange,
  onAdd,
  onCancel,
}) => {
  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? onCancel() : undefined)}
    >
      <DialogContent className="max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Add blocked keyword</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-6">
          <div>
            <Label className="font-bold">Keyword</Label>
            <Input
              placeholder="Enter sensitive keyword or phrase"
              value={keyword}
              onChange={(e) => onKeywordChange(e.target.value)}
              className="mt-2"
            />
          </div>

          <div>
            <Label className="font-bold">Action</Label>
            <p className="text-sm text-muted-foreground mt-1 mb-2">
              Choose what action the guardrail should take when this keyword is
              detected
            </p>
            <Select
              value={action}
              onValueChange={(v) => onActionChange(v as "BLOCK" | "MASK")}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="BLOCK">Block</SelectItem>
                <SelectItem value="MASK">Mask</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label className="font-bold">Description (optional)</Label>
            <Textarea
              placeholder="Explain why this keyword is sensitive"
              value={description}
              onChange={(e) => onDescriptionChange(e.target.value)}
              rows={3}
              className="mt-2"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={onAdd}>Add</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default KeywordModal;

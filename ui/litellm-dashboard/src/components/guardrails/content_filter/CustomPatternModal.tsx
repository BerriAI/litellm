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

interface CustomPatternModalProps {
  visible: boolean;
  patternName: string;
  patternRegex: string;
  patternAction: "BLOCK" | "MASK";
  onNameChange: (name: string) => void;
  onRegexChange: (regex: string) => void;
  onActionChange: (action: "BLOCK" | "MASK") => void;
  onAdd: () => void;
  onCancel: () => void;
}

const CustomPatternModal: React.FC<CustomPatternModalProps> = ({
  visible,
  patternName,
  patternRegex,
  patternAction,
  onNameChange,
  onRegexChange,
  onActionChange,
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
          <DialogTitle>Add custom regex pattern</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-6">
          <div>
            <Label className="font-bold">Pattern name</Label>
            <Input
              placeholder="e.g., internal_id, employee_code"
              value={patternName}
              onChange={(e) => onNameChange(e.target.value)}
              className="mt-2"
            />
          </div>

          <div>
            <Label className="font-bold">Regex pattern</Label>
            <Input
              placeholder="e.g., ID-[0-9]{6}"
              value={patternRegex}
              onChange={(e) => onRegexChange(e.target.value)}
              className="mt-2"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Enter a valid regular expression to match sensitive data
            </p>
          </div>

          <div>
            <Label className="font-bold">Action</Label>
            <p className="text-sm text-muted-foreground mt-1 mb-2">
              Choose what action the guardrail should take when this pattern is
              detected
            </p>
            <Select
              value={patternAction}
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

export default CustomPatternModal;

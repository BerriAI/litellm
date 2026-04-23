import React from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface PrebuiltPattern {
  name: string;
  display_name: string;
  category: string;
  description: string;
}

interface PatternModalProps {
  visible: boolean;
  prebuiltPatterns: PrebuiltPattern[];
  categories: string[];
  selectedPatternName: string;
  patternAction: "BLOCK" | "MASK";
  onPatternNameChange: (name: string) => void;
  onActionChange: (action: "BLOCK" | "MASK") => void;
  onAdd: () => void;
  onCancel: () => void;
}

const PatternModal: React.FC<PatternModalProps> = ({
  visible,
  prebuiltPatterns,
  categories,
  selectedPatternName,
  patternAction,
  onPatternNameChange,
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
          <DialogTitle>Add prebuilt pattern</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-6">
          <div>
            <Label className="font-bold">Pattern type</Label>
            <Select
              value={selectedPatternName}
              onValueChange={onPatternNameChange}
            >
              <SelectTrigger className="w-full mt-2">
                <SelectValue placeholder="Choose pattern type" />
              </SelectTrigger>
              <SelectContent>
                {categories.map((category) => {
                  const categoryPatterns = prebuiltPatterns.filter(
                    (p) => p.category === category,
                  );
                  if (categoryPatterns.length === 0) return null;

                  return (
                    <SelectGroup key={category}>
                      <SelectLabel>{category}</SelectLabel>
                      {categoryPatterns.map((pattern) => (
                        <SelectItem key={pattern.name} value={pattern.name}>
                          {pattern.display_name || pattern.name}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  );
                })}
              </SelectContent>
            </Select>
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

export default PatternModal;

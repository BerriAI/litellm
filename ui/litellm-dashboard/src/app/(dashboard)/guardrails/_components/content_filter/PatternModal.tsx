import React from "react";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxCollection,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxGroup,
  ComboboxInput,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
} from "@/components/ui/combobox";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ACTION_ITEMS } from "./action_options";

interface PrebuiltPattern {
  name: string;
  display_name: string;
  category: string;
  description: string;
}

interface PatternGroup {
  category: string;
  items: PrebuiltPattern[];
}

const matchesPatternQuery = (pattern: PrebuiltPattern, query: string) => {
  const needle = query.toLowerCase();
  return pattern.display_name.toLowerCase().includes(needle) || pattern.name.toLowerCase().includes(needle);
};

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
  const selectedPattern = prebuiltPatterns.find((pattern) => pattern.name === selectedPatternName) ?? null;
  const patternGroups = categories
    .map((category) => ({
      category,
      items: prebuiltPatterns.filter((pattern) => pattern.category === category),
    }))
    .filter((group) => group.items.length > 0);

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Add prebuilt pattern</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          <div>
            <p className="font-semibold">Pattern type</p>
            <Combobox
              items={patternGroups}
              value={selectedPattern}
              onValueChange={(pattern: PrebuiltPattern | null) => pattern && onPatternNameChange(pattern.name)}
              itemToStringLabel={(pattern: PrebuiltPattern) => pattern.display_name}
              filter={matchesPatternQuery}
            >
              <ComboboxInput className="mt-2 w-full" placeholder="Choose pattern type" />
              <ComboboxContent>
                <ComboboxEmpty>No matching patterns</ComboboxEmpty>
                <ComboboxList>
                  {(group: PatternGroup) => (
                    <ComboboxGroup key={group.category} items={group.items}>
                      <ComboboxLabel>{group.category}</ComboboxLabel>
                      <ComboboxCollection>
                        {(pattern: PrebuiltPattern) => (
                          <ComboboxItem key={pattern.name} value={pattern}>
                            {pattern.display_name}
                          </ComboboxItem>
                        )}
                      </ComboboxCollection>
                    </ComboboxGroup>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </div>

          <div>
            <p className="font-semibold">Action</p>
            <p className="mt-1 mb-2 text-muted-foreground">
              Choose what action the guardrail should take when this pattern is detected
            </p>
            <Select
              items={ACTION_ITEMS}
              value={patternAction}
              onValueChange={(value: string | null) => value && onActionChange(value as "BLOCK" | "MASK")}
            >
              <SelectTrigger className="w-full" aria-label="Action">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ACTION_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
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

/**
 * Form component for selecting and configuring fallback groups
 * Manages groups state internally, but does not handle submission
 * Decoupled from form submission logic
 */

import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Plus, X } from "lucide-react";
import React, { useEffect, useState } from "react";
import { toast } from "@/lib/toast";
import { FallbackGroup, FallbackGroupConfig } from "./FallbackGroupConfig";

interface FallbackSelectionFormProps {
  groups: FallbackGroup[];
  onGroupsChange: (groups: FallbackGroup[]) => void;
  availableModels: string[];
  maxFallbacks?: number;
  maxGroups?: number;
}

export function FallbackSelectionForm({
  groups,
  onGroupsChange,
  availableModels,
  maxFallbacks = 10,
  maxGroups = 5,
}: FallbackSelectionFormProps) {
  const [activeKey, setActiveKey] = useState(groups.length > 0 ? groups[0].id : "1");

  // Reset activeKey when groups change (e.g., when modal reopens)
  useEffect(() => {
    if (groups.length > 0) {
      // If current activeKey doesn't exist in groups, reset to first group
      const activeKeyExists = groups.some((g) => g.id === activeKey);
      if (!activeKeyExists) {
        setActiveKey(groups[0].id);
      }
    } else {
      // If groups is empty, reset activeKey
      setActiveKey("1");
    }
  }, [groups]);

  const handleAddGroup = () => {
    if (groups.length >= maxGroups) {
      return;
    }
    const newId = Date.now().toString();
    const newGroups = [
      ...groups,
      {
        id: newId,
        primaryModel: null,
        fallbackModels: [],
      },
    ];
    onGroupsChange(newGroups);
    setActiveKey(newId);
  };

  const handleRemoveGroup = (targetId: string) => {
    if (groups.length === 1) {
      toast.warning("At least one group is required");
      return;
    }
    const newGroups = groups.filter((g) => g.id !== targetId);
    onGroupsChange(newGroups);
    if (activeKey === targetId && newGroups.length > 0) {
      setActiveKey(newGroups[newGroups.length - 1].id);
    }
  };

  const handleGroupUpdate = (updatedGroup: FallbackGroup) => {
    const newGroups = groups.map((g) => (g.id === updatedGroup.id ? updatedGroup : g));
    onGroupsChange(newGroups);
  };

  const groupLabel = (group: FallbackGroup, index: number) =>
    group.primaryModel ? group.primaryModel : `Group ${index + 1}`;

  if (groups.length === 0) {
    return (
      <div className="text-center py-12 bg-muted rounded-lg border border-dashed border-border">
        <p className="text-muted-foreground mb-4">No fallback groups configured</p>
        <Button onClick={handleAddGroup}>
          <Plus className="w-4 h-4" />
          Create First Group
        </Button>
      </div>
    );
  }

  return (
    <Tabs value={activeKey} onValueChange={setActiveKey}>
      <div className="flex items-center border-b">
        <TabsList variant="line" className="h-auto justify-start rounded-none p-0">
          {groups.map((group, index) => (
            <div key={group.id} className="relative flex items-center">
              <TabsTrigger
                value={group.id}
                className={`flex-none rounded-none py-2 pl-4 ${groups.length > 1 ? "pr-9" : "pr-4"}`}
              >
                {groupLabel(group, index)}
              </TabsTrigger>
              {groups.length > 1 && (
                <Button
                  variant="ghost"
                  size="icon-xs"
                  className="absolute right-1"
                  aria-label={`Remove ${groupLabel(group, index)}`}
                  onClick={() => handleRemoveGroup(group.id)}
                >
                  <X />
                </Button>
              )}
            </div>
          ))}
        </TabsList>
        {groups.length < maxGroups && (
          <Button variant="ghost" size="icon-sm" aria-label="Add fallback group" onClick={handleAddGroup}>
            <Plus />
          </Button>
        )}
      </div>
      {groups.map((group) => (
        <TabsContent key={group.id} value={group.id} className="pt-4">
          <FallbackGroupConfig
            group={group}
            onChange={handleGroupUpdate}
            availableModels={availableModels}
            maxFallbacks={maxFallbacks}
          />
        </TabsContent>
      ))}
    </Tabs>
  );
}

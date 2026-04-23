/**
 * Form component for selecting and configuring fallback groups.
 * Manages groups state internally, but does not handle submission.
 */

import { Button } from "@/components/ui/button";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Plus, X } from "lucide-react";
import React, { useEffect, useState } from "react";
import MessageManager from "@/components/molecules/message_manager";
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
  const [activeKey, setActiveKey] = useState(
    groups.length > 0 ? groups[0].id : "1",
  );

  useEffect(() => {
    if (groups.length > 0) {
      const exists = groups.some((g) => g.id === activeKey);
      if (!exists) setActiveKey(groups[0].id);
    } else {
      setActiveKey("1");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groups]);

  const handleAddGroup = () => {
    if (groups.length >= maxGroups) return;
    const newId = Date.now().toString();
    const newGroups = [
      ...groups,
      { id: newId, primaryModel: null, fallbackModels: [] },
    ];
    onGroupsChange(newGroups);
    setActiveKey(newId);
  };

  const handleRemoveGroup = (targetId: string) => {
    if (groups.length === 1) {
      MessageManager.warning("At least one group is required");
      return;
    }
    const newGroups = groups.filter((g) => g.id !== targetId);
    onGroupsChange(newGroups);
    if (activeKey === targetId && newGroups.length > 0) {
      setActiveKey(newGroups[newGroups.length - 1].id);
    }
  };

  const handleGroupUpdate = (updatedGroup: FallbackGroup) => {
    const newGroups = groups.map((g) =>
      g.id === updatedGroup.id ? updatedGroup : g,
    );
    onGroupsChange(newGroups);
  };

  if (groups.length === 0) {
    return (
      <div className="text-center py-12 bg-muted rounded-lg border border-dashed border-border">
        <p className="text-muted-foreground mb-4">
          No fallback groups configured
        </p>
        <Button onClick={handleAddGroup}>
          <Plus className="w-4 h-4" />
          Create First Group
        </Button>
      </div>
    );
  }

  return (
    <div data-testid="fallback-selection-form">
      <Tabs value={activeKey} onValueChange={setActiveKey}>
        <div className="flex items-center justify-between mb-2">
          <TabsList>
            {groups.map((group, index) => {
              const label = group.primaryModel || `Group ${index + 1}`;
              return (
                <TabsTrigger key={group.id} value={group.id}>
                  <span className="inline-flex items-center gap-1">
                    {label}
                    {groups.length > 1 && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveGroup(group.id);
                        }}
                        aria-label={`Remove ${label}`}
                        className="inline-flex items-center"
                      >
                        <X size={12} />
                      </button>
                    )}
                  </span>
                </TabsTrigger>
              );
            })}
          </TabsList>
          {groups.length < maxGroups && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={handleAddGroup}
            >
              <Plus className="h-4 w-4" />
              Add group
            </Button>
          )}
        </div>
        {groups.map((group) => (
          <TabsContent key={group.id} value={group.id}>
            <FallbackGroupConfig
              group={group}
              onChange={handleGroupUpdate}
              availableModels={availableModels}
              maxFallbacks={maxFallbacks}
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}

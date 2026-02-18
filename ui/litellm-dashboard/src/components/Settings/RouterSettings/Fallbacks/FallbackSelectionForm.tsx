/**
 * Form component for selecting and configuring fallback groups
 * Manages groups state internally, but does not handle submission
 * Decoupled from form submission logic
 */

import { Button } from "@tremor/react";
import { message, Tabs } from "antd";
import { Plus } from "lucide-react";
import React, { useEffect, useState } from "react";
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
      message.warning("At least one group is required");
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

  // Generate tab items
  const items = groups.map((group, index) => {
    const label = group.primaryModel
      ? group.primaryModel
      : `Group ${index + 1}`;
    return {
      key: group.id,
      label: label,
      closable: groups.length > 1, // Only allow closing if there's more than 1 group
      children: (
        <FallbackGroupConfig
          group={group}
          onChange={handleGroupUpdate}
          availableModels={availableModels}
          maxFallbacks={maxFallbacks}
        />
      ),
    };
  });

  if (groups.length === 0) {
    return (
      <div className="text-center py-12 bg-gray-50 rounded-lg border border-dashed border-gray-300">
        <p className="text-gray-500 mb-4">No fallback groups configured</p>
        <Button
          variant="primary"
          onClick={handleAddGroup}
          icon={() => <Plus className="w-4 h-4" />}
        >
          Create First Group
        </Button>
      </div>
    );
  }

  return (
    <Tabs
      type="editable-card"
      activeKey={activeKey}
      onChange={setActiveKey}
      onEdit={(targetKey, action) => {
        if (action === "add") handleAddGroup();
        else if (action === "remove" && groups.length > 1) {
          handleRemoveGroup(targetKey as string);
        }
      }}
      items={items}
      className="fallback-tabs"
      tabBarStyle={{
        marginBottom: 0,
      }}
      hideAdd={groups.length >= maxGroups}
    />
  );
}

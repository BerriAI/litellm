import React, { useState, useCallback } from "react";
import { Card, Title, Text, Table, TableHead, TableRow, TableHeaderCell, TableCell, TableBody } from "@tremor/react";
import { Input } from "antd";
import { ChevronDownIcon, ChevronRightIcon, PlusCircleIcon } from "@heroicons/react/outline";
import NotificationManager from "./molecules/notifications_manager";

interface KeyValueItem {
  id?: string;
  key: string;
  value: string;
}

interface GenericKeyValueManagerProps {
  title: string;
  description: string;
  keyLabel: string;
  valueLabel: string;
  keyPlaceholder: string;
  valuePlaceholder: string;
  items: KeyValueItem[];
  onItemsChange: (items: KeyValueItem[]) => void;
  onSave?: () => Promise<void>;
  showSaveButton?: boolean;
  isCollapsible?: boolean;
  defaultExpanded?: boolean;
  configExample?: React.ReactNode;
  additionalActions?: (item: KeyValueItem) => React.ReactNode;
}

const GenericKeyValueManager: React.FC<GenericKeyValueManagerProps> = ({
  title,
  description,
  keyLabel,
  valueLabel,
  keyPlaceholder,
  valuePlaceholder,
  items,
  onItemsChange,
  onSave,
  showSaveButton = true,
  isCollapsible = false,
  defaultExpanded = true,
  configExample,
  additionalActions,
}) => {
  const [newKey, setNewKey] = useState<string>("");
  const [newValue, setNewValue] = useState<string>("");
  const [editingItem, setEditingItem] = useState<KeyValueItem | null>(null);
  const [editingKey, setEditingKey] = useState<string>("");
  const [editingValue, setEditingValue] = useState<string>("");
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const generateId = () => Math.random().toString(36).substr(2, 9);

  const handleAddItem = useCallback(() => {
    if (newKey.trim() && newValue.trim()) {
      const newItem: KeyValueItem = {
        id: generateId(),
        key: newKey.trim(),
        value: newValue.trim(),
      };
      onItemsChange([...items, newItem]);
      setNewKey("");
      setNewValue("");
    } else {
      NotificationManager.fromBackend(`Please provide both ${keyLabel.toLowerCase()} and ${valueLabel.toLowerCase()}`);
    }
  }, [newKey, newValue, items, onItemsChange, keyLabel, valueLabel]);

  const handleEditItem = useCallback((item: KeyValueItem) => {
    setEditingItem({ ...item });
    setEditingKey(item.key);
    setEditingValue(item.value);
  }, []);

  const handleSaveEdit = useCallback(() => {
    if (editingKey.trim() && editingValue.trim()) {
      const updatedItems = items.map((item) =>
        item.id === editingItem?.id ? { ...item, key: editingKey.trim(), value: editingValue.trim() } : item,
      );
      onItemsChange(updatedItems);
      setEditingItem(null);
      setEditingKey("");
      setEditingValue("");
    } else {
      NotificationManager.fromBackend(`Please provide both ${keyLabel.toLowerCase()} and ${valueLabel.toLowerCase()}`);
    }
  }, [editingKey, editingValue, items, editingItem, onItemsChange, keyLabel, valueLabel]);

  const handleCancelEdit = useCallback(() => {
    setEditingItem(null);
    setEditingKey("");
    setEditingValue("");
  }, []);

  const handleDeleteItem = useCallback(
    (id: string) => {
      const updatedItems = items.filter((item) => item.id !== id);
      onItemsChange(updatedItems);
    },
    [items, onItemsChange],
  );

  const handleSave = useCallback(async () => {
    if (onSave) {
      try {
        await onSave();
      } catch (error) {
        console.error("Failed to save:", error);
      }
    }
  }, [onSave]);

  const ContentSection = useCallback(
    () => (
      <div className="space-y-6">
        {/* Add New Item Section */}
        <Card>
          <Title className="mb-4">Add New {keyLabel}</Title>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1">{keyLabel}</label>
              <Input
                value={newKey}
                onChange={(e) => setNewKey(e.target.value)}
                placeholder={keyPlaceholder}
                size="middle"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">{valueLabel}</label>
              <Input
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                placeholder={valuePlaceholder}
                size="middle"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={handleAddItem}
                disabled={!newKey.trim() || !newValue.trim()}
                className={`flex items-center px-4 py-2 rounded-md text-sm ${
                  !newKey.trim() || !newValue.trim()
                    ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                    : "bg-green-600 text-white hover:bg-green-700"
                }`}
              >
                <PlusCircleIcon className="w-4 h-4 mr-1" />
                Add {keyLabel}
              </button>
            </div>
          </div>
        </Card>

        {/* Manage Existing Items Section */}
        <Card>
          <div className="flex justify-between items-center mb-4">
            <Title>Manage Existing {keyLabel}s</Title>
            {showSaveButton && (
              <button onClick={handleSave} className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                Save All Changes
              </button>
            )}
          </div>

          <div className="rounded-lg custom-border relative">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1">
                <TableHead>
                  <TableRow>
                    <TableHeaderCell className="py-1 h-8">{keyLabel}</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">{valueLabel}</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Actions</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {items.map((item) => (
                    <TableRow key={item.id} className="h-8">
                      {editingItem && editingItem.id === item.id ? (
                        <>
                          <TableCell className="py-0.5">
                            <Input value={editingKey} onChange={(e) => setEditingKey(e.target.value)} size="small" />
                          </TableCell>
                          <TableCell className="py-0.5">
                            <Input
                              value={editingValue}
                              onChange={(e) => setEditingValue(e.target.value)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={handleSaveEdit}
                                className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100"
                              >
                                Save
                              </button>
                              <button
                                onClick={handleCancelEdit}
                                className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded hover:bg-gray-100"
                              >
                                Cancel
                              </button>
                            </div>
                          </TableCell>
                        </>
                      ) : (
                        <>
                          <TableCell className="py-0.5 text-sm text-gray-900">{item.key}</TableCell>
                          <TableCell className="py-0.5 text-sm text-gray-500">{item.value}</TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              {additionalActions && additionalActions(item)}
                              <button
                                onClick={() => handleEditItem(item)}
                                className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100"
                              >
                                Edit
                              </button>
                              <button
                                onClick={() => handleDeleteItem(item.id!)}
                                className="text-xs bg-red-50 text-red-600 px-2 py-1 rounded hover:bg-red-100"
                              >
                                Delete
                              </button>
                            </div>
                          </TableCell>
                        </>
                      )}
                    </TableRow>
                  ))}
                  {items.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} className="py-0.5 text-sm text-gray-500 text-center">
                        No {keyLabel.toLowerCase()}s added yet. Add a new {keyLabel.toLowerCase()} above.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </Card>

        {/* Configuration Example */}
        {configExample && (
          <Card>
            <Title className="mb-4">Configuration Example</Title>
            {configExample}
          </Card>
        )}
      </div>
    ),
    [
      keyLabel,
      valueLabel,
      keyPlaceholder,
      valuePlaceholder,
      newKey,
      newValue,
      items,
      editingItem,
      editingKey,
      editingValue,
      showSaveButton,
      configExample,
      additionalActions,
      handleAddItem,
      handleSave,
      handleEditItem,
      handleSaveEdit,
      handleCancelEdit,
      handleDeleteItem,
    ],
  );

  if (isCollapsible) {
    return (
      <Card className="mb-6">
        <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
          <div className="flex flex-col">
            <Title className="mb-0">{title}</Title>
            <p className="text-sm text-gray-500">{description}</p>
          </div>
          <div className="flex items-center">
            {isExpanded ? (
              <ChevronDownIcon className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronRightIcon className="w-5 h-5 text-gray-500" />
            )}
          </div>
        </div>

        {isExpanded && (
          <div className="mt-4">
            <ContentSection />
          </div>
        )}
      </Card>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <Title>{title}</Title>
        <Text className="text-gray-600 mt-2 block">{description}</Text>
      </div>
      <div>
        <ContentSection />
      </div>
    </div>
  );
};

export default GenericKeyValueManager;

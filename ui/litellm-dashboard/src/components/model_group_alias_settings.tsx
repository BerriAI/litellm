import React, { useState, useEffect } from "react";
import { PlusCircleIcon, PencilIcon, TrashIcon, ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { setCallbacksCall } from "./networking";
import { Card, Title, Text, Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";
import NotificationsManager from "./molecules/notifications_manager";

type ModelGroupAliasValue = string | { model: string; hidden?: boolean };

interface ModelGroupAliasSettingsProps {
  accessToken: string;
  initialModelGroupAlias?: Record<string, ModelGroupAliasValue>;
  onAliasUpdate?: (updatedAlias: { [key: string]: string }) => void;
}

interface AliasItem {
  id: string;
  aliasName: string;
  targetModelGroup: string;
}

const ModelGroupAliasSettings: React.FC<ModelGroupAliasSettingsProps> = ({
  accessToken,
  initialModelGroupAlias = {},
  onAliasUpdate,
}) => {
  const [aliases, setAliases] = useState<AliasItem[]>([]);
  const [newAlias, setNewAlias] = useState({ aliasName: "", targetModelGroup: "" });
  const [editingAlias, setEditingAlias] = useState<AliasItem | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const aliasArray = Object.entries(initialModelGroupAlias).map(([aliasName, value], index) => ({
      id: `${index}-${aliasName}`,
      aliasName,
      // if object, use its model field; otherwise use the string
      targetModelGroup: typeof value === "string" ? value : value?.model ?? "",
    }));
    setAliases(aliasArray);
  }, [initialModelGroupAlias]);

  const saveAliasesToBackend = async (updatedAliases: AliasItem[]) => {
    if (!accessToken) {
      console.error("Access token is missing");
      return false;
    }

    try {
      // Convert array back to object format
      const aliasObject: { [key: string]: string } = {};
      updatedAliases.forEach((alias) => {
        aliasObject[alias.aliasName] = alias.targetModelGroup;
      });

      const payload = {
        router_settings: {
          model_group_alias: aliasObject,
        },
      };

      console.log("Saving model group alias:", aliasObject);
      await setCallbacksCall(accessToken, payload);

      if (onAliasUpdate) {
        onAliasUpdate(aliasObject);
      }

      return true;
    } catch (error) {
      console.error("Failed to save model group alias settings:", error);
      NotificationsManager.fromBackend("Failed to save model group alias settings");
      return false;
    }
  };

  const handleAddAlias = async () => {
    if (!newAlias.aliasName || !newAlias.targetModelGroup) {
      NotificationsManager.fromBackend("Please provide both alias name and target model group");
      return;
    }

    // Check for duplicate alias names
    if (aliases.some((alias) => alias.aliasName === newAlias.aliasName)) {
      NotificationsManager.fromBackend("An alias with this name already exists");
      return;
    }

    const newAliasObj: AliasItem = {
      id: `${Date.now()}-${newAlias.aliasName}`,
      aliasName: newAlias.aliasName,
      targetModelGroup: newAlias.targetModelGroup,
    };

    const updatedAliases = [...aliases, newAliasObj];

    if (await saveAliasesToBackend(updatedAliases)) {
      setAliases(updatedAliases);
      setNewAlias({ aliasName: "", targetModelGroup: "" });
      NotificationsManager.success("Alias added successfully");
    }
  };

  const handleEditAlias = (alias: AliasItem) => {
    setEditingAlias({ ...alias });
  };

  const handleUpdateAlias = async () => {
    if (!editingAlias) return;

    if (!editingAlias.aliasName || !editingAlias.targetModelGroup) {
      NotificationsManager.fromBackend("Please provide both alias name and target model group");
      return;
    }

    // Check for duplicate alias names (excluding current alias)
    if (aliases.some((alias) => alias.id !== editingAlias.id && alias.aliasName === editingAlias.aliasName)) {
      NotificationsManager.fromBackend("An alias with this name already exists");
      return;
    }

    const updatedAliases = aliases.map((alias) => (alias.id === editingAlias.id ? editingAlias : alias));

    if (await saveAliasesToBackend(updatedAliases)) {
      setAliases(updatedAliases);
      setEditingAlias(null);
      NotificationsManager.success("Alias updated successfully");
    }
  };

  const handleCancelEdit = () => {
    setEditingAlias(null);
  };

  const deleteAlias = async (aliasId: string) => {
    const updatedAliases = aliases.filter((alias) => alias.id !== aliasId);

    if (await saveAliasesToBackend(updatedAliases)) {
      setAliases(updatedAliases);
      NotificationsManager.success("Alias deleted successfully");
    }
  };

  // Convert current aliases to object for config example
  const aliasObject = aliases.reduce(
    (acc, alias) => {
      acc[alias.aliasName] = alias.targetModelGroup;
      return acc;
    },
    {} as { [key: string]: string },
  );

  return (
    <Card className="mb-6">
      <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex flex-col">
          <Title className="mb-0">Model Group Alias Settings</Title>
          <p className="text-sm text-gray-500">
            Create aliases for your model groups to simplify API calls. For example, you can create an alias
            &apos;gpt-4o&apos; that points to &apos;gpt-4o-mini-openai&apos; model group.
          </p>
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
          <div className="mb-6">
            <Text className="text-sm font-medium text-gray-700 mb-2">Add New Alias</Text>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Alias Name</label>
                <input
                  type="text"
                  value={newAlias.aliasName}
                  onChange={(e) =>
                    setNewAlias({
                      ...newAlias,
                      aliasName: e.target.value,
                    })
                  }
                  placeholder="e.g., gpt-4o"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Target Model Group</label>
                <input
                  type="text"
                  value={newAlias.targetModelGroup}
                  onChange={(e) =>
                    setNewAlias({
                      ...newAlias,
                      targetModelGroup: e.target.value,
                    })
                  }
                  placeholder="e.g., gpt-4o-mini-openai"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleAddAlias}
                  disabled={!newAlias.aliasName || !newAlias.targetModelGroup}
                  className={`flex items-center px-4 py-2 rounded-md text-sm ${!newAlias.aliasName || !newAlias.targetModelGroup ? "bg-gray-300 text-gray-500 cursor-not-allowed" : "bg-green-600 text-white hover:bg-green-700"}`}
                >
                  <PlusCircleIcon className="w-4 h-4 mr-1" />
                  Add Alias
                </button>
              </div>
            </div>
          </div>

          <Text className="text-sm font-medium text-gray-700 mb-2">Manage Existing Aliases</Text>
          <div className="rounded-lg custom-border relative mb-6">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1">
                <TableHead>
                  <TableRow>
                    <TableHeaderCell className="py-1 h-8">Alias Name</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Target Model Group</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Actions</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {aliases.map((alias) => (
                    <TableRow key={alias.id} className="h-8">
                      {editingAlias && editingAlias.id === alias.id ? (
                        <>
                          <TableCell className="py-0.5">
                            <input
                              type="text"
                              value={editingAlias.aliasName}
                              onChange={(e) =>
                                setEditingAlias({
                                  ...editingAlias,
                                  aliasName: e.target.value,
                                })
                              }
                              className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell className="py-0.5">
                            <input
                              type="text"
                              value={editingAlias.targetModelGroup}
                              onChange={(e) =>
                                setEditingAlias({
                                  ...editingAlias,
                                  targetModelGroup: e.target.value,
                                })
                              }
                              className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={handleUpdateAlias}
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
                          <TableCell className="py-0.5 text-sm text-gray-900">{alias.aliasName}</TableCell>
                          <TableCell className="py-0.5 text-sm text-gray-500">{alias.targetModelGroup}</TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={() => handleEditAlias(alias)}
                                className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100"
                              >
                                <PencilIcon className="w-3 h-3" />
                              </button>
                              <button
                                onClick={() => deleteAlias(alias.id)}
                                className="text-xs bg-red-50 text-red-600 px-2 py-1 rounded hover:bg-red-100"
                              >
                                <TrashIcon className="w-3 h-3" />
                              </button>
                            </div>
                          </TableCell>
                        </>
                      )}
                    </TableRow>
                  ))}
                  {aliases.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} className="py-0.5 text-sm text-gray-500 text-center">
                        No aliases added yet. Add a new alias above.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Configuration Example */}
          <Card>
            <Title className="mb-4">Configuration Example</Title>
            <Text className="text-gray-600 mb-4">
              Here&apos;s how your current aliases would look in the config.yaml:
            </Text>
            <div className="bg-gray-100 rounded-lg p-4 font-mono text-sm">
              <div className="text-gray-700">
                router_settings:
                <br />
                &nbsp;&nbsp;model_group_alias:
                {Object.keys(aliasObject).length === 0 ? (
                  <span className="text-gray-500">
                    <br />
                    &nbsp;&nbsp;&nbsp;&nbsp;# No aliases configured yet
                  </span>
                ) : (
                  Object.entries(aliasObject).map(([key, value]) => (
                    <span key={key}>
                      <br />
                      &nbsp;&nbsp;&nbsp;&nbsp;&quot;{key}&quot;: &quot;{value}&quot;
                    </span>
                  ))
                )}
              </div>
            </div>
          </Card>
        </div>
      )}
    </Card>
  );
};

export default ModelGroupAliasSettings;

import React, { useState, useEffect } from "react";
import {
  ChevronDown,
  ChevronRight,
  PlusCircle,
  Pencil,
  Trash2,
} from "lucide-react";
import { setCallbacksCall } from "./networking";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
    <Card className="mb-6 p-6">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex flex-col">
          <h3 className="text-lg font-semibold mb-0">
            Model Group Alias Settings
          </h3>
          <p className="text-sm text-muted-foreground">
            Create aliases for your model groups to simplify API calls. For
            example, you can create an alias &apos;gpt-4o&apos; that points
            to &apos;gpt-4o-mini-openai&apos; model group.
          </p>
        </div>
        <div className="flex items-center">
          {isExpanded ? (
            <ChevronDown className="w-5 h-5 text-muted-foreground" />
          ) : (
            <ChevronRight className="w-5 h-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4">
          <div className="mb-6">
            <span className="block text-sm font-medium text-foreground mb-2">
              Add New Alias
            </span>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">
                  Alias Name
                </label>
                <input
                  type="text"
                  value={newAlias.aliasName}
                  onChange={(e) =>
                    setNewAlias({ ...newAlias, aliasName: e.target.value })
                  }
                  placeholder="e.g., gpt-4o"
                  className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">
                  Target Model Group
                </label>
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
                  className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleAddAlias}
                  disabled={
                    !newAlias.aliasName || !newAlias.targetModelGroup
                  }
                  className={`flex items-center px-4 py-2 rounded-md text-sm ${
                    !newAlias.aliasName || !newAlias.targetModelGroup
                      ? "bg-muted text-muted-foreground cursor-not-allowed"
                      : "bg-emerald-600 text-white hover:bg-emerald-700"
                  }`}
                >
                  <PlusCircle className="w-4 h-4 mr-1" />
                  Add Alias
                </button>
              </div>
            </div>
          </div>

          <span className="block text-sm font-medium text-foreground mb-2">
            Manage Existing Aliases
          </span>
          <div className="rounded-lg custom-border relative mb-6">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1">
                <TableHeader>
                  <TableRow>
                    <TableHead className="py-1 h-8">Alias Name</TableHead>
                    <TableHead className="py-1 h-8">Target Model Group</TableHead>
                    <TableHead className="py-1 h-8">Actions</TableHead>
                  </TableRow>
                </TableHeader>
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
                              className="w-full px-2 py-1 border border-input bg-background rounded-md text-sm"
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
                              className="w-full px-2 py-1 border border-input bg-background rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={handleUpdateAlias}
                                className="text-xs bg-primary/10 text-primary px-2 py-1 rounded hover:bg-primary/20"
                              >
                                Save
                              </button>
                              <button
                                onClick={handleCancelEdit}
                                className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded hover:bg-muted/80"
                              >
                                Cancel
                              </button>
                            </div>
                          </TableCell>
                        </>
                      ) : (
                        <>
                          <TableCell className="py-0.5 text-sm text-foreground">
                            {alias.aliasName}
                          </TableCell>
                          <TableCell className="py-0.5 text-sm text-muted-foreground">
                            {alias.targetModelGroup}
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={() => handleEditAlias(alias)}
                                className="text-xs bg-primary/10 text-primary px-2 py-1 rounded hover:bg-primary/20"
                                aria-label="Edit alias"
                              >
                                <Pencil className="w-3 h-3" />
                              </button>
                              <button
                                onClick={() => deleteAlias(alias.id)}
                                className="text-xs bg-destructive/10 text-destructive px-2 py-1 rounded hover:bg-destructive/20"
                                aria-label="Delete alias"
                              >
                                <Trash2 className="w-3 h-3" />
                              </button>
                            </div>
                          </TableCell>
                        </>
                      )}
                    </TableRow>
                  ))}
                  {aliases.length === 0 && (
                    <TableRow>
                      <TableCell
                        colSpan={3}
                        className="py-0.5 text-sm text-muted-foreground text-center"
                      >
                        No aliases added yet. Add a new alias above.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>

          <Card className="p-6">
            <h4 className="text-base font-semibold mb-4">Configuration Example</h4>
            <p className="text-muted-foreground mb-4 text-sm">
              Here&apos;s how your current aliases would look in the
              config.yaml:
            </p>
            <div className="bg-muted rounded-lg p-4 font-mono text-sm">
              <div className="text-foreground">
                router_settings:
                <br />
                &nbsp;&nbsp;model_group_alias:
                {Object.keys(aliasObject).length === 0 ? (
                  <span className="text-muted-foreground">
                    <br />
                    &nbsp;&nbsp;&nbsp;&nbsp;# No aliases configured yet
                  </span>
                ) : (
                  Object.entries(aliasObject).map(([key, value]) => (
                    <span key={key}>
                      <br />
                      &nbsp;&nbsp;&nbsp;&nbsp;&quot;{key}&quot;: &quot;{value}
                      &quot;
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

import React, { useState, useEffect } from "react";
import { Pencil, Plus, Trash2 } from "lucide-react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import ModelSelector from "./ModelSelector";
import NotificationsManager from "../molecules/notifications_manager";

interface ModelAliasManagerProps {
  accessToken: string;
  initialModelAliases?: { [key: string]: string };
  onAliasUpdate?: (updatedAliases: { [key: string]: string }) => void;
  showExampleConfig?: boolean;
}

interface AliasItem {
  id: string;
  aliasName: string;
  targetModel: string;
}

const ModelAliasManager: React.FC<ModelAliasManagerProps> = ({
  accessToken,
  initialModelAliases = {},
  onAliasUpdate,
  showExampleConfig = true,
}) => {
  const [aliases, setAliases] = useState<AliasItem[]>([]);
  const [newAlias, setNewAlias] = useState({ aliasName: "", targetModel: "" });
  const [editingAlias, setEditingAlias] = useState<AliasItem | null>(null);

  useEffect(() => {
    const aliasArray = Object.entries(initialModelAliases).map(
      ([aliasName, targetModel], index) => ({
        id: `${index}-${aliasName}`,
        aliasName,
        targetModel,
      }),
    );
    setAliases(aliasArray);
  }, [initialModelAliases]);

  const notifyParent = (list: AliasItem[]) => {
    const obj: { [key: string]: string } = {};
    list.forEach((a) => (obj[a.aliasName] = a.targetModel));
    if (onAliasUpdate) onAliasUpdate(obj);
  };

  const handleAddAlias = () => {
    if (!newAlias.aliasName || !newAlias.targetModel) {
      NotificationsManager.fromBackend(
        "Please provide both alias name and target model",
      );
      return;
    }

    if (aliases.some((alias) => alias.aliasName === newAlias.aliasName)) {
      NotificationsManager.fromBackend(
        "An alias with this name already exists",
      );
      return;
    }

    const newAliasObj: AliasItem = {
      id: `${Date.now()}-${newAlias.aliasName}`,
      aliasName: newAlias.aliasName,
      targetModel: newAlias.targetModel,
    };

    const updatedAliases = [...aliases, newAliasObj];
    setAliases(updatedAliases);
    setNewAlias({ aliasName: "", targetModel: "" });
    notifyParent(updatedAliases);

    NotificationsManager.success("Alias added successfully");
  };

  const handleEditAlias = (alias: AliasItem) => {
    setEditingAlias({ ...alias });
  };

  const handleUpdateAlias = () => {
    if (!editingAlias) return;

    if (!editingAlias.aliasName || !editingAlias.targetModel) {
      NotificationsManager.fromBackend(
        "Please provide both alias name and target model",
      );
      return;
    }

    if (
      aliases.some(
        (alias) =>
          alias.id !== editingAlias.id &&
          alias.aliasName === editingAlias.aliasName,
      )
    ) {
      NotificationsManager.fromBackend(
        "An alias with this name already exists",
      );
      return;
    }

    const updatedAliases = aliases.map((alias) =>
      alias.id === editingAlias.id ? editingAlias : alias,
    );

    setAliases(updatedAliases);
    setEditingAlias(null);
    notifyParent(updatedAliases);

    NotificationsManager.success("Alias updated successfully");
  };

  const handleCancelEdit = () => {
    setEditingAlias(null);
  };

  const deleteAlias = (aliasId: string) => {
    const updatedAliases = aliases.filter((alias) => alias.id !== aliasId);
    setAliases(updatedAliases);
    notifyParent(updatedAliases);

    NotificationsManager.success("Alias deleted successfully");
  };

  const aliasObject = aliases.reduce(
    (acc, alias) => {
      acc[alias.aliasName] = alias.targetModel;
      return acc;
    },
    {} as { [key: string]: string },
  );

  const addDisabled = !newAlias.aliasName || !newAlias.targetModel;

  return (
    <div className="mt-4">
      <div className="mb-6">
        <p className="text-sm font-medium text-foreground mb-2">Add New Alias</p>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Alias Name
            </label>
            <Input
              type="text"
              value={newAlias.aliasName}
              onChange={(e) =>
                setNewAlias({
                  ...newAlias,
                  aliasName: e.target.value,
                })
              }
              placeholder="e.g., gpt-4o"
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Target Model
            </label>
            <ModelSelector
              accessToken={accessToken}
              value={newAlias.targetModel}
              placeholder="Select target model"
              onChange={(value) =>
                setNewAlias({
                  ...newAlias,
                  targetModel: value,
                })
              }
              showLabel={false}
            />
          </div>
          <div className="flex items-end">
            <Button
              onClick={handleAddAlias}
              disabled={addDisabled}
              className={cn(
                !addDisabled &&
                  "bg-emerald-600 text-white hover:bg-emerald-700",
              )}
            >
              <Plus className="h-4 w-4" />
              Add Alias
            </Button>
          </div>
        </div>
      </div>

      <p className="text-sm font-medium text-foreground mb-2">
        Manage Existing Aliases
      </p>
      <div className="rounded-lg custom-border relative mb-6">
        <div className="overflow-x-auto">
          <Table className="[&_td]:py-0.5 [&_th]:py-1">
            <TableHead>
              <TableRow>
                <TableHeaderCell className="py-1 h-8">
                  Alias Name
                </TableHeaderCell>
                <TableHeaderCell className="py-1 h-8">
                  Target Model
                </TableHeaderCell>
                <TableHeaderCell className="py-1 h-8">Actions</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {aliases.map((alias) => (
                <TableRow key={alias.id} className="h-8">
                  {editingAlias && editingAlias.id === alias.id ? (
                    <>
                      <TableCell className="py-0.5">
                        <Input
                          type="text"
                          value={editingAlias.aliasName}
                          onChange={(e) =>
                            setEditingAlias({
                              ...editingAlias,
                              aliasName: e.target.value,
                            })
                          }
                          className="h-8"
                        />
                      </TableCell>
                      <TableCell className="py-0.5">
                        <ModelSelector
                          accessToken={accessToken}
                          value={editingAlias.targetModel}
                          onChange={(value) =>
                            setEditingAlias({
                              ...editingAlias,
                              targetModel: value,
                            })
                          }
                          showLabel={false}
                          style={{ height: "32px" }}
                        />
                      </TableCell>
                      <TableCell className="py-0.5 whitespace-nowrap">
                        <div className="flex space-x-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={handleUpdateAlias}
                            className="h-7"
                          >
                            Save
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={handleCancelEdit}
                            className="h-7"
                          >
                            Cancel
                          </Button>
                        </div>
                      </TableCell>
                    </>
                  ) : (
                    <>
                      <TableCell className="py-0.5 text-sm text-foreground">
                        {alias.aliasName}
                      </TableCell>
                      <TableCell className="py-0.5 text-sm text-muted-foreground">
                        {alias.targetModel}
                      </TableCell>
                      <TableCell className="py-0.5 whitespace-nowrap">
                        <div className="flex space-x-2">
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-7 w-7 text-blue-600 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 dark:text-blue-300"
                            onClick={() => handleEditAlias(alias)}
                            aria-label="Edit alias"
                          >
                            <Pencil className="h-3 w-3" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-7 w-7 text-destructive bg-red-50 hover:bg-red-100 dark:bg-red-950/30 dark:hover:bg-red-950/60"
                            onClick={() => deleteAlias(alias.id)}
                            aria-label="Delete alias"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
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

      {showExampleConfig && (
        <Card className="p-4">
          <h5 className="mb-4 font-semibold">Configuration Example</h5>
          <p className="text-muted-foreground mb-4">
            Here&apos;s how your current aliases would look in the config:
          </p>
          <div className="bg-muted rounded-lg p-4 font-mono text-sm">
            <div className="text-foreground">
              model_aliases:
              {Object.keys(aliasObject).length === 0 ? (
                <span className="text-muted-foreground">
                  <br />
                  &nbsp;&nbsp;# No aliases configured yet
                </span>
              ) : (
                Object.entries(aliasObject).map(([key, value]) => (
                  <span key={key}>
                    <br />
                    &nbsp;&nbsp;&quot;{key}&quot;: &quot;{value}&quot;
                  </span>
                ))
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ModelAliasManager;

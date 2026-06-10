import React, { useState, useEffect } from "react";
import { PlusCircleIcon, PencilIcon, TrashIcon } from "@heroicons/react/outline";
import { Card, Title, Text, Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  const [aliases, setAliases] = useState<AliasItem[]>([]);
  const [newAlias, setNewAlias] = useState({ aliasName: "", targetModel: "" });
  const [editingAlias, setEditingAlias] = useState<AliasItem | null>(null);

  useEffect(() => {
    // Convert object to array for display
    const aliasArray = Object.entries(initialModelAliases).map(([aliasName, targetModel], index) => ({
      id: `${index}-${aliasName}`,
      aliasName,
      targetModel,
    }));
    setAliases(aliasArray);
  }, [initialModelAliases]);

  const handleAddAlias = () => {
    if (!newAlias.aliasName || !newAlias.targetModel) {
      NotificationsManager.fromBackend(t("commonComponents.modelAliasManager.bothRequired"));
      return;
    }

    // Check for duplicate alias names
    if (aliases.some((alias) => alias.aliasName === newAlias.aliasName)) {
      NotificationsManager.fromBackend(t("commonComponents.modelAliasManager.duplicateAlias"));
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

    // Convert array back to object format and notify parent
    const aliasObject: { [key: string]: string } = {};
    updatedAliases.forEach((alias) => {
      aliasObject[alias.aliasName] = alias.targetModel;
    });

    if (onAliasUpdate) {
      onAliasUpdate(aliasObject);
    }

    NotificationsManager.success(t("commonComponents.modelAliasManager.addSuccess"));
  };

  const handleEditAlias = (alias: AliasItem) => {
    setEditingAlias({ ...alias });
  };

  const handleUpdateAlias = () => {
    if (!editingAlias) return;

    if (!editingAlias.aliasName || !editingAlias.targetModel) {
      NotificationsManager.fromBackend(t("commonComponents.modelAliasManager.bothRequired"));
      return;
    }

    // Check for duplicate alias names (excluding current alias)
    if (aliases.some((alias) => alias.id !== editingAlias.id && alias.aliasName === editingAlias.aliasName)) {
      NotificationsManager.fromBackend(t("commonComponents.modelAliasManager.duplicateAlias"));
      return;
    }

    const updatedAliases = aliases.map((alias) => (alias.id === editingAlias.id ? editingAlias : alias));

    setAliases(updatedAliases);
    setEditingAlias(null);

    // Convert array back to object format and notify parent
    const aliasObject: { [key: string]: string } = {};
    updatedAliases.forEach((alias) => {
      aliasObject[alias.aliasName] = alias.targetModel;
    });

    if (onAliasUpdate) {
      onAliasUpdate(aliasObject);
    }

    NotificationsManager.success(t("commonComponents.modelAliasManager.updateSuccess"));
  };

  const handleCancelEdit = () => {
    setEditingAlias(null);
  };

  const deleteAlias = (aliasId: string) => {
    const updatedAliases = aliases.filter((alias) => alias.id !== aliasId);
    setAliases(updatedAliases);

    // Convert array back to object format and notify parent
    const aliasObject: { [key: string]: string } = {};
    updatedAliases.forEach((alias) => {
      aliasObject[alias.aliasName] = alias.targetModel;
    });

    if (onAliasUpdate) {
      onAliasUpdate(aliasObject);
    }

    NotificationsManager.success(t("commonComponents.modelAliasManager.deleteSuccess"));
  };

  // Convert current aliases to object for config example
  const aliasObject = aliases.reduce(
    (acc, alias) => {
      acc[alias.aliasName] = alias.targetModel;
      return acc;
    },
    {} as { [key: string]: string },
  );

  return (
    <div className="mt-4">
      <div className="mb-6">
        <Text className="text-sm font-medium text-gray-700 mb-2">
          {t("commonComponents.modelAliasManager.addNewAlias")}
        </Text>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">
              {t("commonComponents.modelAliasManager.aliasName")}
            </label>
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
            <label className="block text-xs text-gray-500 mb-1">
              {t("commonComponents.modelAliasManager.targetModel")}
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
            <button
              onClick={handleAddAlias}
              disabled={!newAlias.aliasName || !newAlias.targetModel}
              className={`flex items-center px-4 py-2 rounded-md text-sm ${!newAlias.aliasName || !newAlias.targetModel ? "bg-gray-300 text-gray-500 cursor-not-allowed" : "bg-green-600 text-white hover:bg-green-700"}`}
            >
              <PlusCircleIcon className="w-4 h-4 mr-1" />
              {t("commonComponents.modelAliasManager.addAlias")}
            </button>
          </div>
        </div>
      </div>

      <Text className="text-sm font-medium text-gray-700 mb-2">
        {t("commonComponents.modelAliasManager.manageExisting")}
      </Text>
      <div className="rounded-lg custom-border relative mb-6">
        <div className="overflow-x-auto">
          <Table className="[&_td]:py-0.5 [&_th]:py-1">
            <TableHead>
              <TableRow>
                <TableHeaderCell className="py-1 h-8">
                  {t("commonComponents.modelAliasManager.aliasName")}
                </TableHeaderCell>
                <TableHeaderCell className="py-1 h-8">
                  {t("commonComponents.modelAliasManager.targetModel")}
                </TableHeaderCell>
                <TableHeaderCell className="py-1 h-8">{t("common.actions")}</TableHeaderCell>
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
                          <button
                            onClick={handleUpdateAlias}
                            className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100"
                          >
                            {t("common.save")}
                          </button>
                          <button
                            onClick={handleCancelEdit}
                            className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded hover:bg-gray-100"
                          >
                            {t("common.cancel")}
                          </button>
                        </div>
                      </TableCell>
                    </>
                  ) : (
                    <>
                      <TableCell className="py-0.5 text-sm text-gray-900">{alias.aliasName}</TableCell>
                      <TableCell className="py-0.5 text-sm text-gray-500">{alias.targetModel}</TableCell>
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
                    {t("commonComponents.modelAliasManager.noAliases")}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      {/* Configuration Example */}
      {showExampleConfig && (
        <Card>
          <Title className="mb-4">{t("commonComponents.modelAliasManager.configExample")}</Title>
          <Text className="text-gray-600 mb-4">{t("commonComponents.modelAliasManager.configExampleDesc")}</Text>
          <div className="bg-gray-100 rounded-lg p-4 font-mono text-sm">
            <div className="text-gray-700">
              model_aliases:
              {Object.keys(aliasObject).length === 0 ? (
                <span className="text-gray-500">
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

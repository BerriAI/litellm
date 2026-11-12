import React, { useState, useEffect, useId } from "react";
import { useTranslation } from "react-i18next";
import { PlusCircleIcon, PencilIcon, TrashIcon } from "@heroicons/react/outline";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableHeader, TableHead, TableBody, TableRow, TableCell } from "@/components/ui/table";
import ModelSelector from "./ModelSelector";
import { toast } from "@/lib/toast";
import { codeSamples } from "@/i18n/untranslated/codeSamples";

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
  const [newAlias, setNewAlias] = useState<{ aliasName: string; targetModel: string | null }>({
    aliasName: "",
    targetModel: null,
  });
  const [editingAlias, setEditingAlias] = useState<
    (Omit<AliasItem, "targetModel"> & { targetModel: string | null }) | null
  >(null);
  const aliasNameId = useId();

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
      toast.fromError(t("modelAliasManager.toasts.missingFields"));
      return;
    }

    // Check for duplicate alias names
    if (aliases.some((alias) => alias.aliasName === newAlias.aliasName)) {
      toast.fromError(t("modelAliasManager.toasts.duplicate"));
      return;
    }

    const newAliasObj: AliasItem = {
      id: `${Date.now()}-${newAlias.aliasName}`,
      aliasName: newAlias.aliasName,
      targetModel: newAlias.targetModel,
    };

    const updatedAliases = [...aliases, newAliasObj];
    setAliases(updatedAliases);
    setNewAlias({ aliasName: "", targetModel: null });

    // Convert array back to object format and notify parent
    const aliasObject: { [key: string]: string } = {};
    updatedAliases.forEach((alias) => {
      aliasObject[alias.aliasName] = alias.targetModel;
    });

    if (onAliasUpdate) {
      onAliasUpdate(aliasObject);
    }

    toast.success(t("modelAliasManager.toasts.added"));
  };

  const handleEditAlias = (alias: AliasItem) => {
    setEditingAlias({ ...alias });
  };

  const handleUpdateAlias = () => {
    if (!editingAlias) return;

    if (!editingAlias.aliasName || !editingAlias.targetModel) {
      toast.fromError(t("modelAliasManager.toasts.missingFields"));
      return;
    }

    // Check for duplicate alias names (excluding current alias)
    if (aliases.some((alias) => alias.id !== editingAlias.id && alias.aliasName === editingAlias.aliasName)) {
      toast.fromError(t("modelAliasManager.toasts.duplicate"));
      return;
    }

    const savedAlias: AliasItem = { ...editingAlias, targetModel: editingAlias.targetModel };
    const updatedAliases = aliases.map((alias) => (alias.id === savedAlias.id ? savedAlias : alias));

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

    toast.success(t("modelAliasManager.toasts.updated"));
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

    toast.success(t("modelAliasManager.toasts.deleted"));
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
        <p className="mb-2 text-sm font-medium text-foreground">{t("modelAliasManager.addTitle")}</p>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label htmlFor={aliasNameId} className="mb-1 block text-xs text-muted-foreground">
              {t("modelAliasManager.aliasName")}
            </label>
            <Input
              id={aliasNameId}
              type="text"
              value={newAlias.aliasName}
              onChange={(e) =>
                setNewAlias({
                  ...newAlias,
                  aliasName: e.target.value,
                })
              }
              placeholder={t("modelAliasManager.aliasPlaceholder")}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-muted-foreground">{t("modelAliasManager.targetModel")}</label>
            <ModelSelector
              accessToken={accessToken}
              value={newAlias.targetModel}
              placeholder={t("modelAliasManager.targetPlaceholder")}
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
            <Button onClick={handleAddAlias} disabled={!newAlias.aliasName || !newAlias.targetModel}>
              <PlusCircleIcon className="mr-1 h-4 w-4" />
              {t("modelAliasManager.addAlias")}
            </Button>
          </div>
        </div>
      </div>

      <p className="mb-2 text-sm font-medium text-foreground">{t("modelAliasManager.manageExistingTitle")}</p>
      <div className="relative mb-6 rounded-lg border">
        <div className="overflow-x-auto">
          <Table className="[&_td]:py-0.5 [&_th]:py-1">
            <TableHeader>
              <TableRow>
                <TableHead className="py-1 h-8">{t("modelAliasManager.columns.aliasName")}</TableHead>
                <TableHead className="py-1 h-8">{t("modelAliasManager.columns.targetModel")}</TableHead>
                <TableHead className="py-1 h-8">{t("modelAliasManager.columns.actions")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {aliases.map((alias) => (
                <TableRow key={alias.id} className="h-8">
                  {editingAlias && editingAlias.id === alias.id ? (
                    <>
                      <TableCell className="py-0.5">
                        <Input
                          type="text"
                          aria-label={t("modelAliasManager.editAliasNameAria")}
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
                          <Button variant="secondary" size="xs" onClick={handleUpdateAlias}>
                            {t("modelAliasManager.save")}
                          </Button>
                          <Button variant="outline" size="xs" onClick={handleCancelEdit}>
                            {t("modelAliasManager.cancel")}
                          </Button>
                        </div>
                      </TableCell>
                    </>
                  ) : (
                    <>
                      <TableCell className="py-0.5 text-sm text-foreground">{alias.aliasName}</TableCell>
                      <TableCell className="py-0.5 text-sm text-muted-foreground">{alias.targetModel}</TableCell>
                      <TableCell className="py-0.5 whitespace-nowrap">
                        <div className="flex space-x-2">
                          <Button
                            variant="secondary"
                            size="icon-xs"
                            aria-label={t("modelAliasManager.editAria", { name: alias.aliasName })}
                            onClick={() => handleEditAlias(alias)}
                          >
                            <PencilIcon className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="destructive"
                            size="icon-xs"
                            aria-label={t("modelAliasManager.deleteAria", { name: alias.aliasName })}
                            onClick={() => deleteAlias(alias.id)}
                          >
                            <TrashIcon className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>
                    </>
                  )}
                </TableRow>
              ))}
              {aliases.length === 0 && (
                <TableRow>
                  <TableCell colSpan={3} className="py-0.5 text-center text-sm text-muted-foreground">
                    {t("modelAliasManager.empty")}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      {/* Configuration Example */}
      {showExampleConfig && (
        <Card className="px-6">
          <CardTitle className="mb-4">{t("modelAliasManager.configExampleTitle")}</CardTitle>
          <p className="mb-4 text-muted-foreground">{t("modelAliasManager.configExampleDescription")}</p>
          <div className="rounded-lg bg-muted p-4 font-mono text-sm">
            <div className="text-foreground">
              model_aliases:
              {Object.keys(aliasObject).length === 0 ? (
                <span className="text-muted-foreground">
                  <br />
                  {codeSamples.models.noAliasesYamlComment}
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

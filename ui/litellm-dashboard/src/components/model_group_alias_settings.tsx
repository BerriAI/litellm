import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  ColumnDef,
  TableOptions,
  Updater,
  functionalUpdate,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, Pin, PinOff } from "lucide-react";
import { PlusCircleIcon, PencilIcon, TrashIcon, ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { setCallbacksCall } from "./networking";
import { Card, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toast } from "@/lib/toast";
import { readStorage, useStoredValue } from "@/lib/storage";
import { DataTableSortHeader } from "@/components/shared/DataTable/DataTableSortHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  AliasDisplayPreferences,
  AliasItem,
  aliasDisplayStorage,
  DEFAULT_ALIAS_DISPLAY,
  movePinnedAlias,
  orderAliases,
} from "./model_group_alias_ordering";

type ModelGroupAliasValue = string | { model: string; hidden?: boolean };

interface ModelGroupAliasSettingsProps {
  accessToken: string;
  initialModelGroupAlias?: Record<string, ModelGroupAliasValue>;
  onAliasUpdate?: (updatedAlias: { [key: string]: string }) => void;
  managedByConfig?: boolean;
  ownershipLoading?: boolean;
  preferenceScope: string | null;
}

const ALIAS_COLUMNS: ColumnDef<AliasItem>[] = [
  { accessorKey: "aliasName", sortDescFirst: false },
  { accessorKey: "targetModelGroup", sortDescFirst: false },
];

function useAliasDisplayPreferences(preferenceScope: string | null) {
  const preferenceKey = useMemo(() => aliasDisplayStorage(preferenceScope ?? ""), [preferenceScope]);
  const [storedPreferences, storePreferences] = useStoredValue(preferenceKey);
  const [unsavedPreferences, setUnsavedPreferences] = useState<AliasDisplayPreferences | null>(null);
  const unsavedRef = useRef<AliasDisplayPreferences | null>(null);
  const preferences = preferenceScope === null ? DEFAULT_ALIAS_DISPLAY : unsavedPreferences ?? storedPreferences;
  const updatePreferences = (updater: Updater<AliasDisplayPreferences>) => {
    if (preferenceScope === null) return;
    const next = functionalUpdate(updater, unsavedRef.current ?? readStorage(preferenceKey));
    storePreferences(next);
    const saved = JSON.stringify(readStorage(preferenceKey)) === JSON.stringify(next);
    // rebind-ok: Async saves need the latest fallback when browser storage refuses writes.
    unsavedRef.current = saved ? null : next;
    setUnsavedPreferences(unsavedRef.current);
  };
  return { preferences, updatePreferences, unsavedPreferences };
}

const ModelGroupAliasSettings: React.FC<ModelGroupAliasSettingsProps> = ({
  accessToken,
  initialModelGroupAlias = {},
  onAliasUpdate,
  managedByConfig = false,
  ownershipLoading = false,
  preferenceScope,
}) => {
  "use no memo";

  const [aliases, setAliases] = useState<AliasItem[]>([]);
  const [newAlias, setNewAlias] = useState({ aliasName: "", targetModelGroup: "" });
  const [editingAlias, setEditingAlias] = useState<AliasItem | null>(null);
  const writesDisabled = managedByConfig || ownershipLoading;
  const [isExpanded, setIsExpanded] = useState(true);
  const { preferences, updatePreferences, unsavedPreferences } = useAliasDisplayPreferences(preferenceScope);
  const displayAliases = useMemo(() => orderAliases(aliases, preferences), [aliases, preferences]);
  const visiblePins = preferences.pinned.filter((name) => aliases.some((alias) => alias.aliasName === name));
  const sorting = [{ id: preferences.sortBy, desc: preferences.descending }];
  const tableOptions: TableOptions<AliasItem> = {
    data: displayAliases,
    columns: ALIAS_COLUMNS,
    getCoreRowModel: getCoreRowModel(),
    manualSorting: true,
    enableSorting: preferenceScope !== null,
    enableSortingRemoval: false,
    enableMultiSort: false,
    state: { sorting },
    onSortingChange: (updater) => {
      const selected = functionalUpdate(updater, sorting)[0];
      updatePreferences({
        ...preferences,
        sortBy: selected?.id === "targetModelGroup" ? "targetModelGroup" : "aliasName",
        descending: selected?.desc ?? false,
      });
    },
  };
  const table = useReactTable(tableOptions);
  const sortDirection = preferences.descending ? "descending" : "ascending";

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

      await setCallbacksCall(accessToken, payload);

      if (onAliasUpdate) {
        onAliasUpdate(aliasObject);
      }

      return true;
    } catch (error) {
      console.error("Failed to save model group alias settings:", error);
      toast.fromError("Failed to save model group alias settings");
      return false;
    }
  };

  const handleAddAlias = async () => {
    if (writesDisabled) return;

    if (!newAlias.aliasName || !newAlias.targetModelGroup) {
      toast.fromError("Please provide both alias name and target model group");
      return;
    }

    // Check for duplicate alias names
    if (aliases.some((alias) => alias.aliasName === newAlias.aliasName)) {
      toast.fromError("An alias with this name already exists");
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
      toast.success("Alias added successfully");
    }
  };

  const handleEditAlias = (alias: AliasItem) => {
    if (writesDisabled) return;

    setEditingAlias({ ...alias });
  };

  const handleUpdateAlias = async () => {
    if (writesDisabled) return;
    if (!editingAlias) return;

    if (!editingAlias.aliasName || !editingAlias.targetModelGroup) {
      toast.fromError("Please provide both alias name and target model group");
      return;
    }

    // Check for duplicate alias names (excluding current alias)
    if (aliases.some((alias) => alias.id !== editingAlias.id && alias.aliasName === editingAlias.aliasName)) {
      toast.fromError("An alias with this name already exists");
      return;
    }

    const updatedAliases = aliases.map((alias) => (alias.id === editingAlias.id ? editingAlias : alias));
    const originalName = aliases.find((alias) => alias.id === editingAlias.id)?.aliasName;

    if (await saveAliasesToBackend(updatedAliases)) {
      if (originalName !== editingAlias.aliasName) {
        updatePreferences((current) => ({
          ...current,
          pinned: current.pinned.map((name) => (name === originalName ? editingAlias.aliasName : name)),
        }));
      }
      setAliases(updatedAliases);
      setEditingAlias(null);
      toast.success("Alias updated successfully");
    }
  };

  const handleCancelEdit = () => {
    setEditingAlias(null);
  };

  const deleteAlias = async (aliasId: string) => {
    if (writesDisabled) return;

    const updatedAliases = aliases.filter((alias) => alias.id !== aliasId);
    const deletedName = aliases.find((alias) => alias.id === aliasId)?.aliasName;

    if (await saveAliasesToBackend(updatedAliases)) {
      updatePreferences((current) => ({
        ...current,
        pinned: current.pinned.filter((name) => name !== deletedName),
      }));
      setAliases(updatedAliases);
      toast.success("Alias deleted successfully");
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
    <Card className="mb-6 px-6">
      <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex flex-col">
          <CardTitle className="mb-0">Model Group Alias Settings</CardTitle>
          <p className="text-sm text-muted-foreground">
            Create aliases for your model groups to simplify API calls. For example, you can create an alias
            &apos;gpt-4o&apos; that points to &apos;gpt-4o-mini-openai&apos; model group.
          </p>
        </div>
        <div className="flex items-center">
          {isExpanded ? (
            <ChevronDownIcon className="w-5 h-5 text-muted-foreground" />
          ) : (
            <ChevronRightIcon className="w-5 h-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4">
          <div className="mb-6">
            {managedByConfig && (
              <p className="mb-2 text-sm text-muted-foreground">
                These aliases are defined in config.yaml and are read-only here. Edit the config file to change them.
              </p>
            )}
            <p className="text-sm font-medium text-foreground mb-2">Add New Alias</p>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Alias Name</label>
                <input
                  type="text"
                  value={newAlias.aliasName}
                  disabled={writesDisabled}
                  onChange={(e) =>
                    setNewAlias({
                      ...newAlias,
                      aliasName: e.target.value,
                    })
                  }
                  placeholder="e.g., gpt-4o"
                  className="w-full px-3 py-2 border border-border rounded-md text-sm disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Target Model Group</label>
                <input
                  type="text"
                  value={newAlias.targetModelGroup}
                  disabled={writesDisabled}
                  onChange={(e) =>
                    setNewAlias({
                      ...newAlias,
                      targetModelGroup: e.target.value,
                    })
                  }
                  placeholder="e.g., gpt-4o-mini-openai"
                  className="w-full px-3 py-2 border border-border rounded-md text-sm disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleAddAlias}
                  disabled={writesDisabled || !newAlias.aliasName || !newAlias.targetModelGroup}
                  className={`flex items-center px-4 py-2 rounded-md text-sm ${writesDisabled || !newAlias.aliasName || !newAlias.targetModelGroup ? "bg-border text-muted-foreground cursor-not-allowed" : "bg-success text-success-foreground hover:bg-success/80"}`}
                >
                  <PlusCircleIcon className="w-4 h-4 mr-1" />
                  Add Alias
                </button>
              </div>
            </div>
          </div>

          <p className="text-sm font-medium text-foreground mb-2">Manage Existing Aliases</p>
          <div className="mb-3 flex flex-wrap items-end gap-3">
            <div>
              <label htmlFor="alias-last-prefix" className="mb-1 block text-xs text-muted-foreground">
                Place aliases with this prefix last
              </label>
              <Input
                id="alias-last-prefix"
                className="w-64"
                placeholder="Optional, e.g. subscription/"
                value={preferences.lastPrefix}
                disabled={preferenceScope === null}
                onChange={(event) => updatePreferences({ ...preferences, lastPrefix: event.target.value })}
              />
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={preferenceScope === null}
              onClick={() => updatePreferences(DEFAULT_ALIAS_DISPLAY)}
            >
              Reset display order
            </Button>
          </div>
          <p className="mb-3 text-xs text-muted-foreground" role={unsavedPreferences ? "status" : undefined}>
            {unsavedPreferences
              ? "Display order could not be saved in this browser. Your changes apply until you leave this page."
              : "Display preferences are saved for your account in this browser. Pinned aliases stay first in your chosen order."}
          </p>
          <div className="rounded-lg custom-border relative mb-6">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1">
                <TableHeader>
                  <TableRow>
                    <TableHead
                      className="py-1 h-8"
                      aria-sort={preferences.sortBy === "aliasName" ? sortDirection : "none"}
                    >
                      <DataTableSortHeader column={table.getColumn("aliasName")!} title="Alias Name" />
                    </TableHead>
                    <TableHead
                      className="py-1 h-8"
                      aria-sort={preferences.sortBy === "targetModelGroup" ? sortDirection : "none"}
                    >
                      <DataTableSortHeader column={table.getColumn("targetModelGroup")!} title="Target Model Group" />
                    </TableHead>
                    <TableHead className="py-1 h-8">Display order</TableHead>
                    <TableHead className="py-1 h-8">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {displayAliases.map((alias) => (
                    <TableRow key={alias.id} className="h-8">
                      {editingAlias && !writesDisabled && editingAlias.id === alias.id ? (
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
                              className="w-full px-2 py-1 border border-border rounded-md text-sm"
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
                              className="w-full px-2 py-1 border border-border rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell />
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={handleUpdateAlias}
                                className="text-xs bg-info/10 text-info px-2 py-1 rounded-sm hover:bg-info/15"
                              >
                                Save
                              </button>
                              <button
                                onClick={handleCancelEdit}
                                className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded-sm hover:bg-accent"
                              >
                                Cancel
                              </button>
                            </div>
                          </TableCell>
                        </>
                      ) : (
                        <>
                          <TableCell className="py-0.5 text-sm whitespace-normal text-foreground">
                            {alias.aliasName}
                          </TableCell>
                          <TableCell className="py-0.5 text-sm whitespace-normal text-muted-foreground">
                            {alias.targetModelGroup}
                          </TableCell>
                          <TableCell className="py-0.5">
                            <div className="flex items-center gap-1">
                              <Button
                                variant="ghost"
                                size="icon-sm"
                                disabled={preferenceScope === null}
                                aria-label={`${visiblePins.includes(alias.aliasName) ? "Unpin" : "Pin"} alias ${alias.aliasName}`}
                                aria-pressed={visiblePins.includes(alias.aliasName)}
                                onClick={() =>
                                  updatePreferences({
                                    ...preferences,
                                    pinned: visiblePins.includes(alias.aliasName)
                                      ? visiblePins.filter((name) => name !== alias.aliasName)
                                      : [...visiblePins, alias.aliasName],
                                  })
                                }
                              >
                                {visiblePins.includes(alias.aliasName) ? (
                                  <PinOff className="size-3.5 text-primary" />
                                ) : (
                                  <Pin className="size-3.5" />
                                )}
                              </Button>
                              {visiblePins.includes(alias.aliasName) &&
                                ([-1, 1] as const).map((direction) => (
                                  <Button
                                    key={direction}
                                    variant="ghost"
                                    size="icon-sm"
                                    aria-label={`Move ${alias.aliasName} ${direction === -1 ? "up" : "down"}`}
                                    disabled={
                                      visiblePins.indexOf(alias.aliasName) + direction < 0 ||
                                      visiblePins.indexOf(alias.aliasName) + direction >= visiblePins.length
                                    }
                                    onClick={() =>
                                      updatePreferences({
                                        ...preferences,
                                        pinned: movePinnedAlias(visiblePins, alias.aliasName, direction),
                                      })
                                    }
                                  >
                                    {direction === -1 ? (
                                      <ArrowUp className="size-3.5" />
                                    ) : (
                                      <ArrowDown className="size-3.5" />
                                    )}
                                  </Button>
                                ))}
                            </div>
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={() => handleEditAlias(alias)}
                                disabled={writesDisabled}
                                aria-label={`Edit alias ${alias.aliasName}`}
                                className={`text-xs px-2 py-1 rounded-sm ${writesDisabled ? "bg-border text-muted-foreground cursor-not-allowed" : "bg-info/10 text-info hover:bg-info/15"}`}
                              >
                                <PencilIcon className="w-3 h-3" />
                              </button>
                              <button
                                onClick={() => deleteAlias(alias.id)}
                                disabled={writesDisabled}
                                aria-label={`Delete alias ${alias.aliasName}`}
                                className={`text-xs px-2 py-1 rounded-sm ${writesDisabled ? "bg-border text-muted-foreground cursor-not-allowed" : "bg-destructive/10 text-destructive hover:bg-destructive/15"}`}
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
                      <TableCell
                        colSpan={4}
                        className="py-0.5 text-sm whitespace-normal text-muted-foreground text-center"
                      >
                        No aliases added yet. Add a new alias above.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Configuration Example */}
          <Card className="px-6">
            <CardTitle className="mb-4">Configuration Example</CardTitle>
            <p className="text-muted-foreground mb-4">
              Here&apos;s how your current aliases would look in the config.yaml:
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

"use client";

import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Plus, RefreshCw, Search, X } from "lucide-react";
import { useRoutingGroups, useSaveRoutingGroups } from "@/app/(dashboard)/hooks/routingGroups/useRoutingGroups";
import { useRouterFields } from "@/app/(dashboard)/hooks/router/useRouterFields";
import { useModelHub } from "@/app/(dashboard)/hooks/models/useModels";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";
import RoutingGroupsTable from "./RoutingGroupsTable";
import RoutingGroupModal from "./RoutingGroupModal";
import { toast } from "@/lib/toast";
import type { RoutingGroup } from "./types";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const RoutingGroups: React.FC = () => {
  const { data, isLoading, refetch, isFetching } = useRoutingGroups();
  const { data: routerFields } = useRouterFields();
  const { data: modelHub } = useModelHub();
  const { accessToken } = useAuthorized();
  const proxySettings = useProxySettings(accessToken);
  const saveMutation = useSaveRoutingGroups();

  const [searchQuery, setSearchQuery] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerMode, setDrawerMode] = useState<"create" | "edit">("create");
  const [editingGroup, setEditingGroup] = useState<RoutingGroup | null>(null);
  const [deletingGroup, setDeletingGroup] = useState<RoutingGroup | null>(null);

  const groups = data?.routingGroups ?? [];

  const filteredGroups = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return groups;
    return groups.filter(
      (g) =>
        g.group_name.toLowerCase().includes(q) ||
        g.routing_strategy.toLowerCase().includes(q) ||
        g.models.some((m) => m.toLowerCase().includes(q)),
    );
  }, [groups, searchQuery]);

  const availableStrategies = useMemo(() => {
    if (data?.availableStrategies?.length) return data.availableStrategies;
    const fromFields = routerFields?.fields?.find((f) => f.field_name === "routing_strategy")?.options;
    return fromFields ?? [];
  }, [data?.availableStrategies, routerFields]);

  const strategyDescriptions = routerFields?.routing_strategy_descriptions ?? {};

  const modelOptions = useMemo<string[]>(() => {
    const records = (modelHub?.data ?? []) as Array<{ model_group?: string }>;
    const names = records.map((r) => r.model_group).filter((n): n is string => Boolean(n));
    return Array.from(new Set(names));
  }, [modelHub]);

  const openCreate = () => {
    setDrawerMode("create");
    setEditingGroup(null);
    setDrawerOpen(true);
  };

  const openEdit = (group: RoutingGroup) => {
    setDrawerMode("edit");
    setEditingGroup(group);
    setDrawerOpen(true);
  };

  const handleSubmit = async (incoming: RoutingGroup) => {
    const next: RoutingGroup[] =
      drawerMode === "create"
        ? [...groups, incoming]
        : groups.map((g) => (g.group_name === editingGroup?.group_name ? incoming : g));

    try {
      await saveMutation.mutateAsync(next);
      toast.success(
        drawerMode === "create"
          ? `Created routing group "${incoming.group_name}"`
          : `Updated routing group "${incoming.group_name}"`,
      );
      setDrawerOpen(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save routing group");
    }
  };

  const confirmDelete = async () => {
    if (!deletingGroup) return;
    const next = groups.filter((g) => g.group_name !== deletingGroup.group_name);
    try {
      await saveMutation.mutateAsync(next);
      toast.success(`Deleted routing group "${deletingGroup.group_name}"`);
      setDeletingGroup(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete routing group");
    }
  };

  return (
    <div className="flex w-full flex-col gap-4">
      <Card size="sm">
        <CardContent>
          <div className="mb-4 flex items-center justify-between gap-3">
            <InputGroup className="max-w-sm">
              <InputGroupAddon>
                <Search className="size-4 text-muted-foreground" />
              </InputGroupAddon>
              <InputGroupInput
                placeholder="Search groups..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              {searchQuery && (
                <InputGroupAddon align="inline-end">
                  <InputGroupButton size="icon-xs" aria-label="Clear search" onClick={() => setSearchQuery("")}>
                    <X />
                  </InputGroupButton>
                </InputGroupAddon>
              )}
            </InputGroup>
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                onClick={() => refetch()}
                disabled={isFetching && !isLoading}
                aria-busy={isFetching && !isLoading}
              >
                <RefreshCw />
                Refresh
              </Button>
              <Button onClick={openCreate}>
                <Plus />
                Create Group
              </Button>
              <span className="text-sm whitespace-nowrap text-muted-foreground">
                Showing {filteredGroups.length} {filteredGroups.length === 1 ? "result" : "results"}
              </span>
            </div>
          </div>

          <RoutingGroupsTable
            groups={filteredGroups}
            isLoading={isLoading}
            onEdit={openEdit}
            onDelete={(g) => setDeletingGroup(g)}
            proxyBaseUrl={proxySettings.LITELLM_UI_API_DOC_BASE_URL?.trim() || proxySettings.PROXY_BASE_URL || ""}
          />
        </CardContent>
      </Card>

      <RoutingGroupModal
        open={drawerOpen}
        mode={drawerMode}
        initialValue={editingGroup}
        availableStrategies={availableStrategies}
        strategyDescriptions={strategyDescriptions}
        modelOptions={modelOptions}
        existingGroupNames={groups.map((g) => g.group_name)}
        onClose={() => setDrawerOpen(false)}
        onSubmit={handleSubmit}
        saving={saveMutation.isPending}
      />

      <Dialog open={Boolean(deletingGroup)} onOpenChange={(open) => !open && setDeletingGroup(null)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Delete routing group?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-foreground">
            Models in <span className="font-medium">{deletingGroup?.group_name}</span> will fall back to the
            proxy&apos;s top-level routing strategy. This cannot be undone.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeletingGroup(null)}>
              Cancel
            </Button>
            <Button
              onClick={confirmDelete}
              variant="destructive"
              disabled={saveMutation.isPending}
              aria-busy={saveMutation.isPending}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default RoutingGroups;

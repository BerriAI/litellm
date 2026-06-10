"use client";

import React, { useMemo, useState } from "react";
import { Button, Card, Flex, Input, Modal, Space, Typography } from "antd";
import { PlusOutlined, ReloadOutlined, SearchOutlined } from "@ant-design/icons";
import { Trans, useTranslation } from "react-i18next";
import { useRoutingGroups, useSaveRoutingGroups } from "@/app/(dashboard)/hooks/routingGroups/useRoutingGroups";
import { useRouterFields } from "@/app/(dashboard)/hooks/router/useRouterFields";
import { useModelHub } from "@/app/(dashboard)/hooks/models/useModels";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";
import RoutingGroupsTable from "./RoutingGroupsTable";
import RoutingGroupModal from "./RoutingGroupModal";
import NotificationsManager from "../molecules/notifications_manager";
import type { RoutingGroup } from "./types";

const { Text } = Typography;

const RoutingGroups: React.FC = () => {
  const { t } = useTranslation();
  const { data, isLoading, refetch, isFetching } = useRoutingGroups();
  const { data: routerFields } = useRouterFields();
  const { data: modelHub } = useModelHub();
  const proxySettings = useProxySettings();
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
      NotificationsManager.success(
        drawerMode === "create"
          ? t("routingGroups.index.createSucceeded", { groupName: incoming.group_name })
          : t("routingGroups.index.updateSucceeded", { groupName: incoming.group_name }),
      );
      setDrawerOpen(false);
    } catch (err) {
      NotificationsManager.error(err instanceof Error ? err.message : t("routingGroups.index.saveFailed"));
    }
  };

  const confirmDelete = async () => {
    if (!deletingGroup) return;
    const next = groups.filter((g) => g.group_name !== deletingGroup.group_name);
    try {
      await saveMutation.mutateAsync(next);
      NotificationsManager.success(t("routingGroups.index.deleteSucceeded", { groupName: deletingGroup.group_name }));
      setDeletingGroup(null);
    } catch (err) {
      NotificationsManager.error(err instanceof Error ? err.message : t("routingGroups.index.deleteFailed"));
    }
  };

  return (
    <Space direction="vertical" size={16} className="w-full">
      <Card bodyStyle={{ padding: 16 }}>
        <Flex justify="space-between" align="center" gap={12} className="mb-4">
          <Input
            allowClear
            prefix={<SearchOutlined className="text-gray-400" />}
            placeholder={t("routingGroups.index.searchPlaceholder")}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="max-w-sm"
          />
          <Flex align="center" gap={12}>
            <Button icon={<ReloadOutlined />} onClick={() => refetch()} loading={isFetching && !isLoading}>
              {t("common.refresh")}
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
              {t("routingGroups.index.createGroup")}
            </Button>
            <Text type="secondary" className="text-sm whitespace-nowrap">
              {t("routingGroups.index.showingResults", { count: filteredGroups.length })}
            </Text>
          </Flex>
        </Flex>

        <RoutingGroupsTable
          groups={filteredGroups}
          loading={isLoading}
          onEdit={openEdit}
          onDelete={(g) => setDeletingGroup(g)}
          proxyBaseUrl={proxySettings.LITELLM_UI_API_DOC_BASE_URL?.trim() || proxySettings.PROXY_BASE_URL || ""}
        />
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

      <Modal
        open={Boolean(deletingGroup)}
        title={t("routingGroups.index.deleteTitle")}
        okText={t("common.delete")}
        okButtonProps={{ danger: true, loading: saveMutation.isPending }}
        cancelText={t("common.cancel")}
        onOk={confirmDelete}
        onCancel={() => setDeletingGroup(null)}
      >
        <Text>
          <Trans
            i18nKey="routingGroups.index.deleteFallbackNote"
            values={{ groupName: deletingGroup?.group_name }}
            components={{ strong: <Text strong /> }}
          />
        </Text>
      </Modal>
    </Space>
  );
};

export default RoutingGroups;

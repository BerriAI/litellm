"use client";

import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { PaginationState } from "@tanstack/react-table";
import { PlusOutlined } from "@ant-design/icons";
import { Button, Space, Typography, message } from "antd";
import React, { useCallback, useMemo, useState } from "react";

import { MemoryRow, createMemory, deleteMemory, fetchMemoryList, updateMemory } from "@/components/networking";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

import { MemoryDetailDrawer } from "./MemoryDetailDrawer";
import { MemoryEditModal } from "./MemoryEditModal";
import { MemoryTable } from "./MemoryTable";

const { Text, Paragraph, Title } = Typography;

interface MemoryViewProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

const DEFAULT_PAGE_SIZE = 50;

export const MemoryView: React.FC<MemoryViewProps> = ({ accessToken }) => {
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: DEFAULT_PAGE_SIZE });
  const [detailRow, setDetailRow] = useState<MemoryRow | null>(null);
  const [editRow, setEditRow] = useState<MemoryRow | null>(null);
  const [deleteRow, setDeleteRow] = useState<MemoryRow | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  const queryClient = useQueryClient();
  // React Query key prefix for all memory-list variants (paged + filtered).
  // Mutations invalidate the whole prefix so the next render refetches the
  // currently-visible page without us needing a manual refetch().
  const MEMORY_LIST_KEY = "memoryList" as const;

  const { data, isLoading, isFetching } = useQuery({
    queryKey: [MEMORY_LIST_KEY, debouncedSearch, pagination.pageIndex, pagination.pageSize],
    queryFn: () => {
      if (!accessToken) throw new Error("Access token required");
      // Prefix search matches the Redis-style mental model (namespace scan):
      // typing "user:" finds "user:profile", "user:prefs", etc.
      return fetchMemoryList(accessToken, {
        keyPrefix: debouncedSearch || undefined,
        page: pagination.pageIndex + 1,
        pageSize: pagination.pageSize,
      });
    },
    enabled: !!accessToken,
  });

  const rows = useMemo(() => data?.memories ?? [], [data]);
  const total = data?.total ?? 0;

  // -- Mutations --------------------------------------------------------
  // All three write endpoints share the same success/error plumbing:
  //   - on success: invalidate the list query so every cached page
  //     refetches from scratch (pagination + filter-aware).
  //   - on error: surface the message via antd `message.error`.

  const invalidateList = useCallback(
    () => queryClient.invalidateQueries({ queryKey: [MEMORY_LIST_KEY] }),
    [queryClient],
  );

  const createMutation = useMutation({
    mutationFn: (args: { key: string; value: string; metadata: unknown }) => {
      if (!accessToken) throw new Error("Access token required");
      return createMemory(accessToken, args);
    },
    onSuccess: (row) => {
      message.success(`Created ${row.key}`);
      invalidateList();
    },
    onError: (err: Error) => {
      message.error(`Save failed: ${err.message}`);
    },
  });

  const updateMutation = useMutation({
    mutationFn: (args: { key: string; value?: string; metadata: unknown }) => {
      if (!accessToken) throw new Error("Access token required");
      const { key, ...payload } = args;
      return updateMemory(accessToken, key, payload);
    },
    onSuccess: (row) => {
      message.success(`Updated ${row.key}`);
      invalidateList();
    },
    onError: (err: Error) => {
      message.error(`Save failed: ${err.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (key: string) => {
      if (!accessToken) throw new Error("Access token required");
      return deleteMemory(accessToken, key).then(() => key);
    },
    onSuccess: (key) => {
      message.success(`Deleted ${key}`);
      invalidateList();
    },
    onError: (err: Error) => {
      message.error(`Delete failed: ${err.message}`);
    },
  });

  const handleSearchChange = useCallback((value: string) => {
    setSearchInput(value);
    setPagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, []);

  const handleView = useCallback((row: MemoryRow) => setDetailRow(row), []);
  const handleEdit = useCallback((row: MemoryRow) => setEditRow(row), []);
  const handleDelete = useCallback((row: MemoryRow) => setDeleteRow(row), []);

  const confirmDelete = async () => {
    if (!deleteRow) return;
    try {
      await deleteMutation.mutateAsync(deleteRow.key);
      setDeleteRow(null);
    } catch {
      // Error toast already surfaced by deleteMutation.onError;
      // leave the modal open so the user can retry or cancel.
    }
  };

  const handleSave = async (key: string, value: string, metadataText: string, isCreate: boolean): Promise<boolean> => {
    if (!accessToken) return false;

    // On edit, an empty textarea is a user's intent to CLEAR existing
    // metadata — we must send explicit `null` (not `undefined`), or
    // JSON.stringify drops the field and the backend's model_fields_set
    // won't see it, leaving the stored value untouched.
    //
    // On create, an empty textarea just means "no metadata" — we omit the
    // field so the DB default (NULL) applies and we avoid Prisma's
    // `Json? = None` quirk on create.
    let metadataPayload: unknown;
    if (!metadataText.trim()) {
      metadataPayload = isCreate ? undefined : null;
    } else {
      try {
        metadataPayload = JSON.parse(metadataText);
      } catch {
        message.error("Metadata must be valid JSON (or leave empty).");
        return false;
      }
    }

    try {
      if (isCreate) {
        await createMutation.mutateAsync({
          key,
          value,
          metadata: metadataPayload,
        });
      } else {
        await updateMutation.mutateAsync({
          key,
          value,
          metadata: metadataPayload,
        });
      }
      return true;
    } catch {
      // error already surfaced by mutation's onError handler
      return false;
    }
  };

  return (
    <div className="w-full" style={{ padding: 24 }}>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
          <div>
            <Title level={3} style={{ marginBottom: 4 }}>
              Memory
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Inspect what your agents have stored under <Text code>/v1/memory</Text>. Scoped to memories visible to
              your user / team (admins see all).
            </Paragraph>
          </div>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setIsCreateOpen(true)}>
            New memory
          </Button>
        </div>

        <MemoryTable
          data={rows}
          isLoading={isLoading}
          rowCount={total}
          pagination={pagination}
          onPaginationChange={setPagination}
          searchValue={searchInput}
          onSearchChange={handleSearchChange}
          isRefreshing={isFetching && !isLoading}
          onRefresh={invalidateList}
          hasActiveSearch={!!debouncedSearch}
          onViewClick={handleView}
          onEditClick={handleEdit}
          onDeleteClick={handleDelete}
        />
      </Space>

      {/* Detail drawer */}
      <MemoryDetailDrawer row={detailRow} onClose={() => setDetailRow(null)} />

      {/* Create / edit modal */}
      <MemoryEditModal
        open={isCreateOpen || !!editRow}
        mode={editRow ? "edit" : "create"}
        initialRow={editRow ?? undefined}
        onClose={() => {
          setIsCreateOpen(false);
          setEditRow(null);
        }}
        onSave={handleSave}
      />

      {/* Delete confirmation modal */}
      <DeleteResourceModal
        isOpen={!!deleteRow}
        title="Delete memory"
        message="This action cannot be undone."
        resourceInformationTitle="Memory"
        resourceInformation={
          deleteRow
            ? [
                { label: "Key", value: deleteRow.key, code: true },
                { label: "Memory ID", value: deleteRow.memory_id, code: true },
                { label: "User ID", value: deleteRow.user_id ?? "-", code: true },
                { label: "Team ID", value: deleteRow.team_id ?? "-", code: true },
              ]
            : []
        }
        onCancel={() => {
          if (!deleteMutation.isPending) setDeleteRow(null);
        }}
        onOk={confirmDelete}
        confirmLoading={deleteMutation.isPending}
        requiredConfirmation={deleteRow?.key}
      />
    </div>
  );
};

export default MemoryView;

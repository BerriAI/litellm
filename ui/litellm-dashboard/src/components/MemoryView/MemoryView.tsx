"use client";

import React, { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Button,
  Card,
  Drawer,
  Empty,
  Input,
  Space,
  Table,
  Tooltip,
  Typography,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import {
  MemoryRow,
  createMemory,
  deleteMemory,
  fetchMemoryList,
  updateMemory,
} from "../networking";
import { MemoryEditModal } from "./MemoryEditModal";
import DeleteResourceModal from "../common_components/DeleteResourceModal";

const { Text, Paragraph, Title } = Typography;

interface MemoryViewProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

function previewValue(value: string, max = 120): string {
  if (!value) return "";
  const trimmed = value.trim();
  if (trimmed.length <= max) return trimmed;
  return `${trimmed.slice(0, max)}…`;
}

function formatTimestamp(ts?: string): string {
  if (!ts) return "—";
  try {
    const d = new Date(ts);
    return d.toLocaleString();
  } catch {
    return ts;
  }
}

const PAGE_SIZE = 50;

export const MemoryView: React.FC<MemoryViewProps> = ({ accessToken }) => {
  const [searchInput, setSearchInput] = useState("");
  const [appliedSearch, setAppliedSearch] = useState("");
  const [detailRow, setDetailRow] = useState<MemoryRow | null>(null);
  const [editRow, setEditRow] = useState<MemoryRow | null>(null);
  const [deleteRow, setDeleteRow] = useState<MemoryRow | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  // Reset to page 1 whenever the filter changes.
  React.useEffect(() => {
    setCurrentPage(1);
  }, [appliedSearch]);

  const queryClient = useQueryClient();
  // React Query key prefix for all memory-list variants (paged + filtered).
  // Mutations invalidate the whole prefix so the next render refetches the
  // currently-visible page without us needing a manual refetch().
  const MEMORY_LIST_KEY = "memoryList" as const;

  const {
    data,
    isLoading,
    isFetching,
  } = useQuery({
    queryKey: [MEMORY_LIST_KEY, appliedSearch, currentPage],
    queryFn: () => {
      if (!accessToken) throw new Error("Access token required");
      // Prefix search matches the Redis-style mental model (namespace scan):
      // typing "user:" finds "user:profile", "user:prefs", etc.
      return fetchMemoryList(accessToken, {
        keyPrefix: appliedSearch || undefined,
        page: currentPage,
        pageSize: PAGE_SIZE,
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

  const invalidateList = () =>
    queryClient.invalidateQueries({ queryKey: [MEMORY_LIST_KEY] });

  const createMutation = useMutation({
    mutationFn: (args: {
      key: string;
      value: string;
      metadata: unknown;
    }) => {
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
    mutationFn: (args: {
      key: string;
      value?: string;
      metadata: unknown;
    }) => {
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

  const handleDelete = (row: MemoryRow) => {
    setDeleteRow(row);
  };

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

  const handleSave = async (
    key: string,
    value: string,
    metadataText: string,
    isCreate: boolean,
  ): Promise<boolean> => {
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

  const renderIdPill = (
    id: string | null | undefined,
    onClick?: () => void,
  ) => {
    if (!id) return <Text type="secondary">-</Text>;
    const short = id.length > 10 ? `${id.slice(0, 7)}...` : id;
    const pillClass =
      "font-mono text-blue-600 bg-blue-50 text-xs font-medium px-2 py-0.5 rounded-md border border-blue-200 inline-block max-w-[15ch] truncate whitespace-nowrap";
    return (
      <Tooltip title={id}>
        {onClick ? (
          <button
            onClick={onClick}
            className={`${pillClass} hover:bg-blue-100 cursor-pointer transition-colors text-left`}
          >
            {short}
          </button>
        ) : (
          <span className={pillClass}>{short}</span>
        )}
      </Tooltip>
    );
  };

  const columns: ColumnsType<MemoryRow> = [
    {
      title: "ID",
      dataIndex: "memory_id",
      key: "memory_id",
      width: 140,
      render: (_: unknown, r: MemoryRow) =>
        renderIdPill(r.memory_id, () => setDetailRow(r)),
    },
    {
      title: "Name",
      dataIndex: "key",
      key: "key",
      width: 200,
      render: (k: string) => <Text code>{k}</Text>,
      // No client-side sorter: pagination is server-side, so a client sort
      // would only reorder the current page and mislead users into thinking
      // the whole list is sorted. Backend returns rows ordered by
      // `updated_at DESC`; use the prefix filter for discovery by name.
    },
    {
      title: "Preview",
      dataIndex: "value",
      key: "value",
      render: (v: string) => (
        <Text type="secondary" style={{ whiteSpace: "pre-wrap" }}>
          {previewValue(v)}
        </Text>
      ),
    },
    {
      title: "User ID",
      dataIndex: "user_id",
      key: "user_id",
      width: 160,
      render: (uid?: string | null) => renderIdPill(uid),
    },
    {
      title: "Team ID",
      dataIndex: "team_id",
      key: "team_id",
      width: 160,
      render: (tid?: string | null) => renderIdPill(tid),
    },
    {
      title: "Updated",
      dataIndex: "updated_at",
      key: "updated_at",
      width: 180,
      render: (ts?: string) => <Text type="secondary">{formatTimestamp(ts)}</Text>,
      // No sorter — backend already returns rows in `updated_at DESC` order,
      // and a client-side sorter on a paginated view would only affect the
      // current page.
    },
    {
      title: "",
      key: "actions",
      width: 140,
      render: (_: unknown, r: MemoryRow) => (
        <Space size={4}>
          <Button
            size="small"
            type="text"
            icon={<EyeOutlined />}
            onClick={() => setDetailRow(r)}
            aria-label="View"
          />
          <Button
            size="small"
            type="text"
            icon={<EditOutlined />}
            onClick={() => setEditRow(r)}
            aria-label="Edit"
          />
          <Button
            size="small"
            type="text"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(r)}
            aria-label="Delete"
          />
        </Space>
      ),
    },
  ];

  return (
    <div className="w-full" style={{ padding: 24 }}>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>
            Memory
          </Title>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            Inspect what your agents have stored under <Text code>/v1/memory</Text>.
            Scoped to memories visible to your user / team (admins see all).
          </Paragraph>
        </div>

        <Card>
          <Space
            style={{
              width: "100%",
              justifyContent: "space-between",
              marginBottom: 16,
            }}
            wrap
          >
            <Space>
              <Input
                allowClear
                placeholder='Filter by key prefix, e.g. "user:"'
                prefix={<SearchOutlined />}
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onPressEnter={() => setAppliedSearch(searchInput.trim())}
                onClear={() => {
                  setSearchInput("");
                  setAppliedSearch("");
                }}
                style={{ width: 280 }}
              />
              <Button
                type="primary"
                ghost
                onClick={() => setAppliedSearch(searchInput.trim())}
              >
                Search
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => invalidateList()}
                loading={isFetching && !isLoading}
              >
                Refresh
              </Button>
            </Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setIsCreateOpen(true)}
            >
              New memory
            </Button>
          </Space>

          <Table
            rowKey="memory_id"
            loading={isLoading}
            dataSource={rows}
            columns={columns}
            // Server-side pagination: we fetch one page at a time so we never
            // silently truncate large stores. `total` drives the page count;
            // changing page/pageSize retriggers the query via `currentPage`.
            pagination={{
              current: currentPage,
              pageSize: PAGE_SIZE,
              total,
              showSizeChanger: false,
              showTotal: (n, range) =>
                `${range[0]}–${range[1]} of ${n}`,
              onChange: (page) => setCurrentPage(page),
            }}
            locale={{
              emptyText: (
                <Empty
                  description={
                    appliedSearch
                      ? `No memories with keys starting with "${appliedSearch}"`
                      : "No memories stored yet"
                  }
                />
              ),
            }}
          />
        </Card>
      </Space>

      {/* Detail drawer */}
      <Drawer
        open={!!detailRow}
        onClose={() => setDetailRow(null)}
        title={
          detailRow ? (
            <Space>
              <Text code>{detailRow.key}</Text>
            </Space>
          ) : (
            "Memory"
          )
        }
        width={720}
        destroyOnClose
      >
        {detailRow && (
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            <Space size="large" wrap>
              <div>
                <Text strong style={{ display: "block" }}>
                  Memory ID
                </Text>
                <Text code style={{ fontSize: 12 }}>
                  {detailRow.memory_id}
                </Text>
              </div>
              <div>
                <Text strong style={{ display: "block" }}>
                  User ID
                </Text>
                <Text type={detailRow.user_id ? undefined : "secondary"}>
                  {detailRow.user_id ?? "-"}
                </Text>
              </div>
              <div>
                <Text strong style={{ display: "block" }}>
                  Team ID
                </Text>
                <Text type={detailRow.team_id ? undefined : "secondary"}>
                  {detailRow.team_id ?? "-"}
                </Text>
              </div>
            </Space>
            <div>
              <Text strong>Value</Text>
              <Paragraph
                style={{
                  background: "#fafafa",
                  padding: 12,
                  borderRadius: 6,
                  whiteSpace: "pre-wrap",
                  fontFamily:
                    "ui-monospace, SFMono-Regular, Menlo, monospace",
                  fontSize: 13,
                }}
              >
                {detailRow.value}
              </Paragraph>
            </div>
            {detailRow.metadata !== undefined &&
              detailRow.metadata !== null && (
                <div>
                  <Text strong>Metadata</Text>
                  <Paragraph
                    style={{
                      background: "#fafafa",
                      padding: 12,
                      borderRadius: 6,
                      whiteSpace: "pre-wrap",
                      fontFamily:
                        "ui-monospace, SFMono-Regular, Menlo, monospace",
                      fontSize: 12,
                    }}
                  >
                    {JSON.stringify(detailRow.metadata, null, 2)}
                  </Paragraph>
                </div>
              )}
            <Space
              split={<Text type="secondary">·</Text>}
              wrap
              size="small"
              style={{ color: "rgba(0,0,0,0.45)" }}
            >
              <Text type="secondary">
                Created {formatTimestamp(detailRow.created_at)}
                {detailRow.created_by ? ` by ${detailRow.created_by}` : ""}
              </Text>
              <Text type="secondary">
                Updated {formatTimestamp(detailRow.updated_at)}
                {detailRow.updated_by ? ` by ${detailRow.updated_by}` : ""}
              </Text>
            </Space>
          </Space>
        )}
      </Drawer>

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

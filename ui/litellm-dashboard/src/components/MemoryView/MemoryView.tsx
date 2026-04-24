"use client";

import React, { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Button,
  Card,
  Drawer,
  Empty,
  Input,
  Modal,
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
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  // Reset to page 1 whenever the filter changes.
  React.useEffect(() => {
    setCurrentPage(1);
  }, [appliedSearch]);

  const {
    data,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ["memoryList", appliedSearch, currentPage],
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

  const handleDelete = async (row: MemoryRow) => {
    Modal.confirm({
      title: "Delete memory",
      content: (
        <span>
          Delete memory key <Text code>{row.key}</Text>? This cannot be undone.
        </span>
      ),
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        if (!accessToken) return;
        try {
          await deleteMemory(accessToken, row.key);
          message.success(`Deleted ${row.key}`);
          refetch();
        } catch (err: any) {
          message.error(`Delete failed: ${err?.message ?? err}`);
        }
      },
    });
  };

  const handleSave = async (
    key: string,
    value: string,
    metadataText: string,
    isCreate: boolean,
  ): Promise<boolean> => {
    if (!accessToken) return false;
    let metadataParsed: unknown | undefined;
    if (metadataText.trim()) {
      try {
        metadataParsed = JSON.parse(metadataText);
      } catch {
        message.error("Metadata must be valid JSON (or leave empty).");
        return false;
      }
    }
    try {
      if (isCreate) {
        await createMemory(accessToken, {
          key,
          value,
          metadata: metadataParsed,
        });
        message.success(`Created ${key}`);
      } else {
        await updateMemory(accessToken, key, {
          value,
          metadata: metadataParsed,
        });
        message.success(`Updated ${key}`);
      }
      refetch();
      return true;
    } catch (err: any) {
      message.error(`Save failed: ${err?.message ?? err}`);
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
      sorter: (a, b) => a.key.localeCompare(b.key),
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
      render: (ts?: string) => (
        <Text type="secondary">{formatTimestamp(ts)}</Text>
      ),
      sorter: (a, b) =>
        new Date(a.updated_at ?? 0).getTime() -
        new Date(b.updated_at ?? 0).getTime(),
      defaultSortOrder: "descend",
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
                onClick={() => refetch()}
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
    </div>
  );
};

export default MemoryView;

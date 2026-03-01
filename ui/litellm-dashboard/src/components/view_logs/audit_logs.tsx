import { useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { Table, Tag, Input, Select, Button, Tooltip } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import moment from "moment";
import { uiAuditLogsCall } from "../networking";
import { AuditLogEntry } from "./columns";
import { AuditLogDrawer } from "./AuditLogDrawer/AuditLogDrawer";
import { Team } from "../key_team_helpers/key_list";

const { Search } = Input;

interface AuditLogsProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  isActive: boolean;
  premiumUser: boolean;
  allTeams: Team[];
}

const asset_logos_folder = "../ui/assets/";
export const auditLogsPreviewImg = `${asset_logos_folder}audit-logs-preview.png`;

const TABLE_NAME_DISPLAY: Record<string, string> = {
  LiteLLM_VerificationToken: "Keys",
  LiteLLM_TeamTable: "Teams",
  LiteLLM_UserTable: "Users",
  LiteLLM_OrganizationTable: "Organizations",
  LiteLLM_ProxyModelTable: "Models",
};

const ACTION_COLOR: Record<string, string> = {
  created: "green",
  updated: "blue",
  deleted: "red",
  rotated: "orange",
};

const PAGE_SIZE = 50;

export default function AuditLogs({
  userID,
  userRole,
  token,
  accessToken,
  isActive,
  premiumUser,
}: AuditLogsProps) {
  const [page, setPage] = useState(1);

  // Filter state
  const [objectId, setObjectId] = useState("");
  const [changedBy, setChangedBy] = useState("");
  const [keyHash, setKeyHash] = useState("");
  const [teamId, setTeamId] = useState("");
  const [action, setAction] = useState<string | undefined>(undefined);
  const [tableName, setTableName] = useState<string | undefined>(undefined);

  // Drawer state
  const [selectedLog, setSelectedLog] = useState<AuditLogEntry | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const query = useQuery({
    queryKey: [
      "audit_logs",
      page,
      PAGE_SIZE,
      objectId,
      changedBy,
      keyHash,
      teamId,
      action,
      tableName,
    ],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return { audit_logs: [], total: 0, page: 1, page_size: PAGE_SIZE, total_pages: 0 };
      }
      return uiAuditLogsCall({
        accessToken,
        page,
        page_size: PAGE_SIZE,
        params: {
          object_id: objectId || undefined,
          changed_by: changedBy || undefined,
          object_key_hash: keyHash || undefined,
          object_team_id: teamId || undefined,
          action: action || undefined,
          table_name: tableName || undefined,
          sort_by: "updated_at",
          sort_order: "desc",
        },
      });
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && isActive,
    placeholderData: keepPreviousData,
  });

  const handleFilterChange = () => {
    // Reset to page 1 whenever a filter changes
    setPage(1);
  };

  const handleTableChange = (pagination: TablePaginationConfig) => {
    setPage(pagination.current ?? 1);
  };

  const handleRowClick = (log: AuditLogEntry) => {
    setSelectedLog(log);
    setDrawerOpen(true);
  };

  const columns: ColumnsType<AuditLogEntry> = [
    {
      title: "Timestamp",
      dataIndex: "updated_at",
      key: "updated_at",
      width: 200,
      render: (val: string) => (
        <span className="font-mono text-xs whitespace-nowrap">
          {moment.utc(val).local().format("MMM D, YYYY HH:mm:ss")}
        </span>
      ),
    },
    {
      title: "Action",
      dataIndex: "action",
      key: "action",
      width: 100,
      render: (val: string) => (
        <Tag color={ACTION_COLOR[val] ?? "default"} className="capitalize">
          {val}
        </Tag>
      ),
    },
    {
      title: "Table",
      dataIndex: "table_name",
      key: "table_name",
      width: 130,
      render: (val: string) => TABLE_NAME_DISPLAY[val] ?? val,
    },
    {
      title: "Object ID",
      dataIndex: "object_id",
      key: "object_id",
      render: (val: string) => (
        <Tooltip title={val}>
          <span className="font-mono text-xs">{val}</span>
        </Tooltip>
      ),
    },
    {
      title: "Changed By",
      dataIndex: "changed_by",
      key: "changed_by",
      width: 200,
      render: (val: string) => val || "—",
    },
    {
      title: "API Key",
      dataIndex: "changed_by_api_key",
      key: "changed_by_api_key",
      width: 140,
      render: (val: string) =>
        val ? (
          <Tooltip title={val}>
            <span className="font-mono text-xs">{val.slice(0, 12)}…</span>
          </Tooltip>
        ) : (
          "—"
        ),
    },
  ];

  if (!premiumUser) {
    return (
      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <h1 style={{ display: "block", marginBottom: "10px" }}>✨ Enterprise Feature.</h1>
        <p style={{ display: "block", marginBottom: "10px" }}>
          This is a LiteLLM Enterprise feature, and requires a valid key to use.
        </p>
        <p style={{ display: "block", marginBottom: "20px", fontStyle: "italic" }}>
          Here&apos;s a preview of what Audit Logs offer:
        </p>
        <img
          src={auditLogsPreviewImg}
          alt="Audit Logs Preview"
          style={{
            maxWidth: "100%",
            maxHeight: "700px",
            borderRadius: "8px",
            boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
            margin: "0 auto",
          }}
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>
    );
  }

  const auditLogs: AuditLogEntry[] = query.data?.audit_logs ?? [];
  const total: number = query.data?.total ?? 0;

  return (
    <>
      <div className="bg-white rounded-lg shadow">
        {/* Header */}
        <div className="border-b px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-semibold">Audit Logs</h1>
            <Button
              icon={<ReloadOutlined spin={query.isFetching} />}
              onClick={() => query.refetch()}
              disabled={query.isFetching}
            >
              Refresh
            </Button>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-3">
            <Search
              placeholder="Object ID"
              allowClear
              style={{ width: 200 }}
              onSearch={(val) => {
                setObjectId(val);
                handleFilterChange();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setObjectId("");
                  handleFilterChange();
                }
              }}
            />
            <Search
              placeholder="Changed By"
              allowClear
              style={{ width: 180 }}
              onSearch={(val) => {
                setChangedBy(val);
                handleFilterChange();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setChangedBy("");
                  handleFilterChange();
                }
              }}
            />
            <Search
              placeholder="Team ID"
              allowClear
              style={{ width: 180 }}
              onSearch={(val) => {
                setTeamId(val);
                handleFilterChange();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setTeamId("");
                  handleFilterChange();
                }
              }}
            />
            <Search
              placeholder="Key Hash"
              allowClear
              style={{ width: 180 }}
              onSearch={(val) => {
                setKeyHash(val);
                handleFilterChange();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setKeyHash("");
                  handleFilterChange();
                }
              }}
            />
            <Select
              placeholder="All Actions"
              allowClear
              style={{ width: 140 }}
              options={[
                { label: "Created", value: "created" },
                { label: "Updated", value: "updated" },
                { label: "Deleted", value: "deleted" },
                { label: "Rotated", value: "rotated" },
              ]}
              onChange={(val) => {
                setAction(val);
                handleFilterChange();
              }}
            />
            <Select
              placeholder="All Tables"
              allowClear
              style={{ width: 150 }}
              options={[
                { label: "Keys", value: "LiteLLM_VerificationToken" },
                { label: "Teams", value: "LiteLLM_TeamTable" },
                { label: "Users", value: "LiteLLM_UserTable" },
                { label: "Organizations", value: "LiteLLM_OrganizationTable" },
                { label: "Models", value: "LiteLLM_ProxyModelTable" },
              ]}
              onChange={(val) => {
                setTableName(val);
                handleFilterChange();
              }}
            />
          </div>
        </div>

        {/* Table */}
        <Table<AuditLogEntry>
          columns={columns}
          dataSource={auditLogs}
          rowKey="id"
          loading={query.isLoading}
          size="small"
          onRow={(record) => ({
            onClick: () => handleRowClick(record),
            style: { cursor: "pointer" },
          })}
          pagination={{
            current: page,
            pageSize: PAGE_SIZE,
            total,
            showTotal: (t) => `${t} total`,
            showSizeChanger: false,
            onChange: (p) => setPage(p),
          }}
          onChange={handleTableChange}
        />
      </div>

      <AuditLogDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        log={selectedLog}
      />
    </>
  );
}

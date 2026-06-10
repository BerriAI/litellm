import { useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { Table, Tag, Input, Select, Button, Pagination, Spin } from "antd";
import { ReloadOutlined, LoadingOutlined } from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import moment from "moment";
import { useTranslation } from "react-i18next";
import { uiAuditLogsCall } from "../networking";
import { AuditLogEntry } from "./columns";
import { AuditLogDrawer } from "./AuditLogDrawer/AuditLogDrawer";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";

const { Search } = Input;

const TABLE_NAME_I18N_KEY: Record<string, string> = {
  LiteLLM_VerificationToken: "viewLogs.auditLogs.tableKeys",
  LiteLLM_TeamTable: "viewLogs.auditLogs.tableTeams",
  LiteLLM_UserTable: "viewLogs.auditLogs.tableUsers",
  LiteLLM_OrganizationTable: "viewLogs.auditLogs.tableOrganizations",
  LiteLLM_ProxyModelTable: "viewLogs.auditLogs.tableModels",
};

interface AuditLogsProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  isActive: boolean;
  premiumUser: boolean;
}

const asset_logos_folder = "../ui/assets/";
export const auditLogsPreviewImg = `${asset_logos_folder}audit-logs-preview.png`;

const ACTION_COLOR: Record<string, string> = {
  created: "green",
  updated: "blue",
  deleted: "red",
  rotated: "orange",
};

const PAGE_SIZE = 50;

export default function AuditLogs({ userID, userRole, token, accessToken, isActive, premiumUser }: AuditLogsProps) {
  const { t } = useTranslation();
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
    queryKey: ["audit_logs", page, PAGE_SIZE, objectId, changedBy, keyHash, teamId, action, tableName],
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

  const resetPage = () => setPage(1);

  const handleRowClick = (log: AuditLogEntry) => {
    setSelectedLog(log);
    setDrawerOpen(true);
  };

  const columns: ColumnsType<AuditLogEntry> = [
    {
      title: t("viewLogs.auditLogs.colTimestamp"),
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
      title: t("viewLogs.auditLogs.colAction"),
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
      title: t("viewLogs.auditLogs.colTable"),
      dataIndex: "table_name",
      key: "table_name",
      width: 130,
      render: (val: string) => {
        const key = TABLE_NAME_I18N_KEY[val];
        return key ? t(key) : val;
      },
    },
    {
      title: t("viewLogs.auditLogs.colObjectId"),
      dataIndex: "object_id",
      key: "object_id",
      render: (val: string) => <span className="font-mono text-xs">{val}</span>,
    },
    {
      title: t("viewLogs.auditLogs.colChangedBy"),
      dataIndex: "changed_by",
      key: "changed_by",
      width: 200,
      render: (val: string) => <DefaultProxyAdminTag userId={val} />,
    },
    {
      title: t("viewLogs.auditLogs.colApiKeyHash"),
      dataIndex: "changed_by_api_key",
      key: "changed_by_api_key",
      width: 140,
      render: (val: string) => (val ? <span className="font-mono text-xs">{val.slice(0, 12)}…</span> : "—"),
    },
  ];

  if (!premiumUser) {
    return (
      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <h1 style={{ display: "block", marginBottom: "10px" }}>✨ {t("viewLogs.auditLogs.enterpriseTitle")}</h1>
        <p style={{ display: "block", marginBottom: "10px" }}>{t("viewLogs.auditLogs.enterpriseDesc")}</p>
        <p style={{ display: "block", marginBottom: "20px", fontStyle: "italic" }}>
          {t("viewLogs.auditLogs.previewText")}
        </p>
        <img
          src={auditLogsPreviewImg}
          alt={t("viewLogs.auditLogs.previewAlt")}
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
            <h1 className="text-xl font-semibold">{t("viewLogs.auditLogs.pageTitle")}</h1>
          </div>

          {/* Filters + pagination on same row */}
          <div className="flex flex-wrap items-center gap-3">
            <Search
              placeholder={t("viewLogs.auditLogs.placeholderObjectId")}
              allowClear
              style={{ width: 200 }}
              onSearch={(val) => {
                setObjectId(val);
                resetPage();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setObjectId("");
                  resetPage();
                }
              }}
            />
            <Search
              placeholder={t("viewLogs.auditLogs.placeholderChangedBy")}
              allowClear
              style={{ width: 180 }}
              onSearch={(val) => {
                setChangedBy(val);
                resetPage();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setChangedBy("");
                  resetPage();
                }
              }}
            />
            <Search
              placeholder={t("viewLogs.auditLogs.placeholderTeamId")}
              allowClear
              style={{ width: 180 }}
              onSearch={(val) => {
                setTeamId(val);
                resetPage();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setTeamId("");
                  resetPage();
                }
              }}
            />
            <Search
              placeholder={t("viewLogs.auditLogs.placeholderKeyHash")}
              allowClear
              style={{ width: 180 }}
              onSearch={(val) => {
                setKeyHash(val);
                resetPage();
              }}
              onChange={(e) => {
                if (!e.target.value) {
                  setKeyHash("");
                  resetPage();
                }
              }}
            />
            <Select
              placeholder={t("viewLogs.auditLogs.allActions")}
              allowClear
              style={{ width: 140 }}
              options={[
                { label: t("viewLogs.auditLogs.actionCreated"), value: "created" },
                { label: t("viewLogs.auditLogs.actionUpdated"), value: "updated" },
                { label: t("viewLogs.auditLogs.actionDeleted"), value: "deleted" },
                { label: t("viewLogs.auditLogs.actionRotated"), value: "rotated" },
              ]}
              onChange={(val) => {
                setAction(val);
                resetPage();
              }}
            />
            <Select
              placeholder={t("viewLogs.auditLogs.allTables")}
              allowClear
              style={{ width: 150 }}
              options={[
                { label: t("viewLogs.auditLogs.tableKeys"), value: "LiteLLM_VerificationToken" },
                { label: t("viewLogs.auditLogs.tableTeams"), value: "LiteLLM_TeamTable" },
                { label: t("viewLogs.auditLogs.tableUsers"), value: "LiteLLM_UserTable" },
                { label: t("viewLogs.auditLogs.tableOrganizations"), value: "LiteLLM_OrganizationTable" },
                { label: t("viewLogs.auditLogs.tableModels"), value: "LiteLLM_ProxyModelTable" },
              ]}
              onChange={(val) => {
                setTableName(val);
                resetPage();
              }}
            />

            {/* Pagination + refresh pushed to the right */}
            <div className="ml-auto flex items-center gap-2">
              <Button
                icon={<ReloadOutlined spin={query.isFetching} />}
                onClick={() => query.refetch()}
                disabled={query.isFetching}
              />
              <Pagination
                current={page}
                pageSize={PAGE_SIZE}
                total={total}
                showTotal={(n) => t("viewLogs.auditLogs.showTotal", { total: n })}
                showSizeChanger={false}
                size="small"
                onChange={(p) => setPage(p)}
              />
            </div>
          </div>
        </div>

        {/* Table — pagination handled in header */}
        <Table<AuditLogEntry>
          columns={columns}
          dataSource={auditLogs}
          rowKey="id"
          loading={{
            spinning: query.isLoading,
            indicator: <Spin indicator={<LoadingOutlined spin />} size="small" />,
          }}
          size="small"
          pagination={false}
          onRow={(record) => ({
            onClick: () => handleRowClick(record),
            style: { cursor: "pointer" },
          })}
        />
      </div>

      <AuditLogDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} log={selectedLog} />
    </>
  );
}

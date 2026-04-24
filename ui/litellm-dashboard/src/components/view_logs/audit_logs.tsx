import { useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import moment from "moment";
import { RefreshCcw as ReloadOutlined, LoaderCircle, Search as SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { uiAuditLogsCall } from "../networking";
import { AuditLogEntry } from "./columns";
import { AuditLogDrawer } from "./AuditLogDrawer/AuditLogDrawer";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import { DataTable } from "./table";

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

const TABLE_NAME_DISPLAY: Record<string, string> = {
  LiteLLM_VerificationToken: "Keys",
  LiteLLM_TeamTable: "Teams",
  LiteLLM_UserTable: "Users",
  LiteLLM_OrganizationTable: "Organizations",
  LiteLLM_ProxyModelTable: "Models",
};

/** Per-action palette — categorical; kept raw via eslint override. */
const ACTION_BADGE_CLASS: Record<string, string> = {
  created: "bg-green-100 text-green-700 hover:bg-green-100",
  updated: "bg-blue-100 text-blue-700 hover:bg-blue-100",
  deleted: "bg-red-100 text-red-700 hover:bg-red-100",
  rotated: "bg-orange-100 text-orange-700 hover:bg-orange-100",
};

const PAGE_SIZE = 50;

const ACTION_OPTIONS = [
  { label: "All Actions", value: "__all__" },
  { label: "Created", value: "created" },
  { label: "Updated", value: "updated" },
  { label: "Deleted", value: "deleted" },
  { label: "Rotated", value: "rotated" },
];

const TABLE_OPTIONS = [
  { label: "All Tables", value: "__all__" },
  { label: "Keys", value: "LiteLLM_VerificationToken" },
  { label: "Teams", value: "LiteLLM_TeamTable" },
  { label: "Users", value: "LiteLLM_UserTable" },
  { label: "Organizations", value: "LiteLLM_OrganizationTable" },
  { label: "Models", value: "LiteLLM_ProxyModelTable" },
];

/** Small debounced-on-enter/clear text input used for filter fields. */
function FilterSearchInput({
  placeholder,
  width,
  onSubmit,
  onClear,
}: {
  placeholder: string;
  width: number;
  onSubmit: (val: string) => void;
  onClear: () => void;
}) {
  const [value, setValue] = useState("");
  return (
    <div className="relative" style={{ width }}>
      <SearchIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
      <Input
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          if (!e.target.value) onClear();
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSubmit(value);
        }}
        placeholder={placeholder}
        className="pl-8 pr-8 h-9 text-sm"
      />
      {value && (
        <button
          type="button"
          onClick={() => {
            setValue("");
            onClear();
          }}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-sm"
          aria-label="Clear"
        >
          ×
        </button>
      )}
    </div>
  );
}

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

  const resetPage = () => setPage(1);

  const handleRowClick = (log: AuditLogEntry) => {
    setSelectedLog(log);
    setDrawerOpen(true);
  };

  const columns: ColumnDef<AuditLogEntry>[] = [
    {
      header: "Timestamp",
      accessorKey: "updated_at",
      cell: (info) => (
        <span className="font-mono text-xs whitespace-nowrap">
          {moment.utc(info.getValue() as string).local().format("MMM D, YYYY HH:mm:ss")}
        </span>
      ),
    },
    {
      header: "Action",
      accessorKey: "action",
      cell: (info) => {
        const val = info.getValue() as string;
        return (
          <Badge className={`capitalize ${ACTION_BADGE_CLASS[val] ?? "bg-muted text-foreground hover:bg-muted"}`}>
            {val}
          </Badge>
        );
      },
    },
    {
      header: "Table",
      accessorKey: "table_name",
      cell: (info) => {
        const val = info.getValue() as string;
        return <span>{TABLE_NAME_DISPLAY[val] ?? val}</span>;
      },
    },
    {
      header: "Object ID",
      accessorKey: "object_id",
      cell: (info) => <span className="font-mono text-xs">{info.getValue() as string}</span>,
    },
    {
      header: "Changed By",
      accessorKey: "changed_by",
      cell: (info) => <DefaultProxyAdminTag userId={info.getValue() as string} />,
    },
    {
      header: "API Key (Hash)",
      accessorKey: "changed_by_api_key",
      cell: (info) => {
        const val = info.getValue() as string;
        return val ? (
          <span className="font-mono text-xs">{val.slice(0, 12)}…</span>
        ) : (
          <span>—</span>
        );
      },
    },
  ];

  if (!premiumUser) {
    return (
      <div className="text-center mt-5">
        <h1 className="block mb-2.5 text-xl font-semibold">✨ Enterprise Feature.</h1>
        <p className="block mb-2.5">
          This is a LiteLLM Enterprise feature, and requires a valid key to use.
        </p>
        <p className="block mb-5 italic text-muted-foreground">
          Here&apos;s a preview of what Audit Logs offer:
        </p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={auditLogsPreviewImg}
          alt="Audit Logs Preview"
          className="max-w-full max-h-[700px] rounded-lg shadow-md mx-auto"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>
    );
  }

  const auditLogs: AuditLogEntry[] = query.data?.audit_logs ?? [];
  const total: number = query.data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <>
      <div className="bg-background rounded-lg shadow">
        {/* Header */}
        <div className="border-b px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-semibold">Audit Logs</h1>
          </div>

          {/* Filters + pagination on same row */}
          <div className="flex flex-wrap items-center gap-3">
            <FilterSearchInput
              placeholder="Object ID"
              width={200}
              onSubmit={(val) => { setObjectId(val); resetPage(); }}
              onClear={() => { setObjectId(""); resetPage(); }}
            />
            <FilterSearchInput
              placeholder="Changed By"
              width={180}
              onSubmit={(val) => { setChangedBy(val); resetPage(); }}
              onClear={() => { setChangedBy(""); resetPage(); }}
            />
            <FilterSearchInput
              placeholder="Team ID"
              width={180}
              onSubmit={(val) => { setTeamId(val); resetPage(); }}
              onClear={() => { setTeamId(""); resetPage(); }}
            />
            <FilterSearchInput
              placeholder="Key Hash"
              width={180}
              onSubmit={(val) => { setKeyHash(val); resetPage(); }}
              onClear={() => { setKeyHash(""); resetPage(); }}
            />
            <Select
              value={action ?? "__all__"}
              onValueChange={(val) => {
                setAction(val === "__all__" ? undefined : val);
                resetPage();
              }}
            >
              <SelectTrigger className="h-9 w-[140px] text-sm">
                <SelectValue placeholder="All Actions" />
              </SelectTrigger>
              <SelectContent>
                {ACTION_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select
              value={tableName ?? "__all__"}
              onValueChange={(val) => {
                setTableName(val === "__all__" ? undefined : val);
                resetPage();
              }}
            >
              <SelectTrigger className="h-9 w-[150px] text-sm">
                <SelectValue placeholder="All Tables" />
              </SelectTrigger>
              <SelectContent>
                {TABLE_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Pagination + refresh pushed to the right */}
            <div className="ml-auto flex items-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => query.refetch()}
                disabled={query.isFetching}
                aria-label="Refresh"
                className="h-9 w-9"
              >
                <ReloadOutlined className={`h-4 w-4 ${query.isFetching ? "animate-spin" : ""}`} />
              </Button>
              <div className="flex items-center gap-1 text-sm">
                <span className="text-muted-foreground whitespace-nowrap">{total} total</span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="h-8"
                >
                  Prev
                </Button>
                <span className="whitespace-nowrap">
                  Page {page} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="h-8"
                >
                  Next
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Table — pagination handled in header */}
        {query.isLoading ? (
          <div className="p-6 space-y-2">
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
            <div className="flex items-center justify-center pt-2 text-muted-foreground gap-2">
              <LoaderCircle className="h-4 w-4 animate-spin" />
              <span className="text-sm">Loading audit logs…</span>
            </div>
          </div>
        ) : (
          <DataTable<AuditLogEntry, unknown>
            columns={columns}
            data={auditLogs}
            onRowClick={handleRowClick}
          />
        )}
      </div>

      <AuditLogDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        log={selectedLog}
      />
    </>
  );
}

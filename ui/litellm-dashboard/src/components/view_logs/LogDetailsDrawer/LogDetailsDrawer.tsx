import { useEffect, useMemo, useState } from "react";
import { Button, Drawer } from "antd";
import {
  CheckOutlined,
  CopyOutlined,
  LeftOutlined,
  RightOutlined,
} from "@ant-design/icons";
import { Sparkles, Wrench } from "lucide-react";
import { LogEntry } from "../columns";
import { MCP_CALL_TYPES } from "../constants";
import { getEventDisplayName } from "../utils";
import { DrawerHeader } from "./DrawerHeader";
import { useKeyboardNavigation } from "./useKeyboardNavigation";
import { LogDetailContent, GuardrailJumpLink } from "./LogDetailContent";
import { sessionSpendLogsCall } from "../../networking";
import { useQuery } from "@tanstack/react-query";
import { getSpendString } from "@/utils/dataUtils";
import { normalizeGuardrailEntries } from "./utils";
import { DRAWER_WIDTH } from "./constants";
import { useLogDetails } from "@/app/(dashboard)/hooks/logDetails/useLogDetails";

export interface LogDetailsDrawerProps {
  open: boolean;
  onClose: () => void;
  logEntry: LogEntry | null;
  sessionId?: string | null;
  accessToken?: string | null;
  onOpenSettings?: () => void;
  allLogs?: LogEntry[];
  onSelectLog?: (log: LogEntry) => void;
  startTime?: string;
}

const SIDEBAR_WIDTH_PX = 224;

/* ------------------------------------------------------------------ */
/*  TraceEventRow — compact event row used in both session & non-     */
/*  session sidebar lists.  Extracted to avoid JSX duplication.       */
/* ------------------------------------------------------------------ */
interface TraceEventRowProps {
  row: LogEntry;
  isSelected: boolean;
  onClick: () => void;
}

function TraceEventRow({ row, isSelected, onClick }: TraceEventRowProps) {
  const isMcp = MCP_CALL_TYPES.includes(row.call_type);
  const durationValue =
    row.duration != null
      ? row.duration.toFixed(3)
      : row.startTime && row.endTime
        ? ((Date.parse(row.endTime) - Date.parse(row.startTime)) / 1000).toFixed(3)
        : "-";

  return (
    <button
      type="button"
      className={`w-full text-left pl-8 pr-2 py-1 transition-colors ${
        isSelected ? "bg-blue-50" : "hover:bg-slate-100"
      }`}
      onClick={onClick}
    >
      <div className="flex items-center gap-1">
        {isMcp ? (
          <Wrench size={12} className="text-slate-500 flex-shrink-0" />
        ) : (
          <Sparkles size={12} className="text-slate-500 flex-shrink-0" />
        )}
        <span className="text-xs font-medium text-slate-900 truncate">
          {getEventDisplayName(row.call_type, row.model)}
        </span>
      </div>
      <div className="text-[10px] text-slate-500 mt-0 flex items-center gap-1.5 font-mono">
        <span>{durationValue}s</span>
        {row.spend ? (
          <>
            <span>·</span>
            <span>{getSpendString(row.spend)}</span>
          </>
        ) : null}
        {row.total_tokens ? (
          <>
            <span>·</span>
            <span>{row.total_tokens} tok</span>
          </>
        ) : null}
      </div>
    </button>
  );
}

/**
 * Right-side drawer panel for displaying detailed log information.
 * Features:
 * - Request ID prominently displayed with copy functionality
 * - Keyboard navigation (J/K for next/prev, Escape to close)
 * - Formatted and JSON view toggle for request/response
 * - Smart display of cache fields (hidden when zero)
 * - Error alerts for failed requests
 * - Collapsible sections for guardrails, vector store, metadata
 */
export function LogDetailsDrawer({
  open,
  onClose,
  logEntry,
  sessionId,
  accessToken,
  onOpenSettings,
  allLogs = [],
  onSelectLog,
  startTime,
}: LogDetailsDrawerProps) {
  const isSessionMode = Boolean(sessionId);
  const [selectedSessionRequestId, setSelectedSessionRequestId] = useState<string | null>(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [copiedLeftPanelId, setCopiedLeftPanelId] = useState(false);

  const { data: sessionLogs = [] } = useQuery({
    queryKey: ["sessionLogs", sessionId],
    queryFn: async () => {
      if (!sessionId || !accessToken) return [];
      const response = await sessionSpendLogsCall(accessToken, sessionId);
      const allSessionLogs: LogEntry[] = response.data || response || [];
      return allSessionLogs
        .map((row) => ({
          ...row,
          duration: (Date.parse(row.endTime) - Date.parse(row.startTime)) / 1000,
        }))
        .sort((a, b) => {
          const aIsMcp = MCP_CALL_TYPES.includes(a.call_type) ? 1 : 0;
          const bIsMcp = MCP_CALL_TYPES.includes(b.call_type) ? 1 : 0;
          if (aIsMcp !== bIsMcp) return aIsMcp - bIsMcp;
          return new Date(a.startTime).getTime() - new Date(b.startTime).getTime();
        });
    },
    enabled: Boolean(open && isSessionMode && sessionId && accessToken),
  });

  const currentLog = useMemo(() => {
    if (!isSessionMode) return logEntry;
    if (!sessionLogs.length) return null;
    if (selectedSessionRequestId) {
      return sessionLogs.find((row) => row.request_id === selectedSessionRequestId) || sessionLogs[0];
    }
    if (logEntry?.request_id) {
      const clickedLog = sessionLogs.find((row) => row.request_id === logEntry.request_id);
      return clickedLog || sessionLogs[0];
    }
    return sessionLogs[0];
  }, [isSessionMode, logEntry, selectedSessionRequestId, sessionLogs]);

  useEffect(() => {
    if (!isSessionMode || !sessionLogs.length) return;
    if (!selectedSessionRequestId || !sessionLogs.some((row) => row.request_id === selectedSessionRequestId)) {
      const fallbackRequestId = logEntry?.request_id && sessionLogs.some((row) => row.request_id === logEntry.request_id)
        ? logEntry.request_id
        : sessionLogs[0].request_id;
      setSelectedSessionRequestId(fallbackRequestId);
    }
  }, [isSessionMode, logEntry, selectedSessionRequestId, sessionLogs]);

  // Reset transient UI state when the drawer opens or closes.
  useEffect(() => {
    if (open) {
      setIsSidebarCollapsed(false);
    } else {
      if (isSessionMode) setSelectedSessionRequestId(null);
      setCopiedLeftPanelId(false);
    }
  }, [open, isSessionMode]);

  // Keyboard navigation
  const { selectNextLog, selectPreviousLog } = useKeyboardNavigation({
    isOpen: open,
    currentLog,
    allLogs: isSessionMode ? sessionLogs : allLogs,
    onClose,
    onSelectLog: (selected) => {
      if (isSessionMode) {
        setSelectedSessionRequestId(selected.request_id);
      }
      onSelectLog?.(selected);
    },
  });

  // Lazy-load log details (messages/response) only when drawer is open.
  // This fetches data for a single log on-demand instead of prefetching all 50.
  const logDetails = useLogDetails(currentLog?.request_id, startTime, open && !!currentLog?.request_id);
  const detailsData = logDetails.data as any;
  const isLoadingDetails = logDetails.isLoading;

  // Build an enriched log entry that merges lazy-loaded details.
  // The list endpoint may already include messages/response when store_prompts_in_spend_logs is enabled,
  // while the detail endpoint fetches from custom loggers (S3, GCS, etc.) or DB fallback.
  const enrichedLog = useMemo(() => {
    if (!currentLog) return null;
    return {
      ...currentLog,
      messages: detailsData?.messages || currentLog.messages,
      response: detailsData?.response || currentLog.response,
      proxy_server_request: detailsData?.proxy_server_request || currentLog.proxy_server_request,
    };
  }, [currentLog, detailsData]);

  const metadata = currentLog?.metadata || {};

  // Status display values
  const statusLabel = metadata.status === "failure" ? "Failure" : "Success";
  const statusColor = metadata.status === "failure" ? ("error" as const) : ("success" as const);
  const environment = metadata?.user_api_key_team_alias || "default";

  const totalSessionCost = sessionLogs.reduce((sum, row) => sum + (row.spend || 0), 0);
  const sessionStart = sessionLogs.length > 0
    ? new Date(Math.min(...sessionLogs.map((r) => new Date(r.startTime).getTime())))
    : null;
  const sessionEnd = sessionLogs.length > 0
    ? new Date(Math.max(...sessionLogs.map((r) => new Date(r.endTime).getTime())))
    : null;
  const sessionDurationSeconds =
    sessionStart && sessionEnd ? ((sessionEnd.getTime() - sessionStart.getTime()) / 1000).toFixed(2) : "0.00";
  const llmCount = sessionLogs.filter((row) => !MCP_CALL_TYPES.includes(row.call_type)).length;
  const mcpCount = sessionLogs.filter((row) => MCP_CALL_TYPES.includes(row.call_type)).length;
  const logsForList = isSessionMode ? sessionLogs : currentLog ? [currentLog] : [];
  const leftPanelId = isSessionMode ? sessionId || "" : currentLog?.request_id || "";
  const leftPanelDisplayId =
    leftPanelId.length > 14 ? `${leftPanelId.slice(0, 11)}...` : leftPanelId;

  const handleCopyLeftPanelId = async () => {
    if (!leftPanelId) return;
    try {
      await navigator.clipboard.writeText(leftPanelId);
      setCopiedLeftPanelId(true);
      setTimeout(() => setCopiedLeftPanelId(false), 1200);
    } catch { /* clipboard unavailable in non-secure contexts */ }
  };

  if (!currentLog || !enrichedLog) return null;

  return (
    <Drawer
      title={null}
      placement="right"
      onClose={onClose}
      open={open}
      width={DRAWER_WIDTH}
      closable={false}
      mask={true}
      maskClosable={true}
      styles={{
        body: { padding: 0, overflow: "hidden" },
        header: { display: "none" },
      }}
    >
      <div style={{ height: "100%" }} className="flex relative">
          {!isSidebarCollapsed ? (
            <Button
              type="text"
              size="small"
              icon={<LeftOutlined />}
              onClick={() => setIsSidebarCollapsed(true)}
              className="absolute top-2 left-2 z-20 !bg-white !border !border-slate-200 !rounded-md"
              aria-label="Collapse trace sidebar"
            />
          ) : (
            <Button
              type="text"
              size="small"
              icon={<RightOutlined />}
              onClick={() => setIsSidebarCollapsed(false)}
              className="absolute top-2 left-2 z-20 !bg-white !border !border-slate-200 !rounded-md"
              aria-label="Expand trace sidebar"
            />
          )}
          {!isSidebarCollapsed && (
          <div
            className="border-r border-slate-200 bg-slate-50 flex flex-col"
            style={{ width: SIDEBAR_WIDTH_PX }}
          >
            <div className="pl-12 pr-3 py-2 border-b border-slate-200 bg-white">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-slate-500">
                    {isSessionMode ? "Session" : "Trace"}
                  </div>
                  <div className="font-mono text-[12px] text-slate-900 leading-tight flex items-center gap-1">
                    <span className="truncate">{leftPanelDisplayId}</span>
                    <button
                      type="button"
                      onClick={handleCopyLeftPanelId}
                      className="text-slate-400 hover:text-slate-600"
                      aria-label="Copy trace id"
                    >
                      {copiedLeftPanelId ? (
                        <CheckOutlined className="text-[11px]" />
                      ) : (
                        <CopyOutlined className="text-[11px]" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
              <div className="mt-1 text-[11px] text-slate-500 font-mono">
                {logsForList.length} req
                <span className="mx-1.5">·</span>
                {isSessionMode
                  ? `${llmCount} LLM`
                  : `${logsForList.filter((row) => !MCP_CALL_TYPES.includes(row.call_type)).length} LLM`}
                <span className="mx-1.5">·</span>
                {isSessionMode
                  ? `${mcpCount} MCP`
                  : `${logsForList.filter((row) => MCP_CALL_TYPES.includes(row.call_type)).length} MCP`}
                <span className="mx-1.5">·</span>
                {isSessionMode
                  ? getSpendString(totalSessionCost)
                  : getSpendString(currentLog.spend || 0)}
                {isSessionMode && (
                  <>
                    <span className="mx-1.5">·</span>
                    {sessionDurationSeconds}s
                  </>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto">
              {normalizeGuardrailEntries(metadata?.guardrail_information).length > 0 && (
                <div className="px-3 pt-2">
                  <GuardrailJumpLink guardrailEntries={normalizeGuardrailEntries(metadata?.guardrail_information)} />
                </div>
              )}
              {isSessionMode ? (
                <div className="py-1">
                  {/* Child events — vertical tree line with horizontal connectors */}
                  <div className="relative pl-2">
                    <div className="absolute left-4 top-1 bottom-1 border-l border-slate-300" />
                    {logsForList.map((row, idx) => {
                      const isLast = idx === logsForList.length - 1;
                      return (
                        <div key={row.request_id} className="relative">
                          <div className="absolute left-4 top-3 w-3 border-t border-slate-300" />
                          {isLast && <div className="absolute left-4 top-3 bottom-0 w-px bg-slate-50" />}
                          <TraceEventRow
                            row={row}
                            isSelected={row.request_id === currentLog.request_id}
                            onClick={() => {
                              setSelectedSessionRequestId(row.request_id);
                              onSelectLog?.(row);
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="py-1">
                  {logsForList.map((row) => (
                    <TraceEventRow
                      key={row.request_id}
                      row={row}
                      isSelected={row.request_id === currentLog.request_id}
                      onClick={() => onSelectLog?.(row)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
          )}

          <div className="flex-1 flex flex-col overflow-hidden">
            <DrawerHeader
              log={currentLog}
              onClose={onClose}
              onPrevious={selectPreviousLog}
              onNext={selectNextLog}
              statusLabel={statusLabel}
              statusColor={statusColor}
              environment={environment}
            />
            <div className="flex-1 overflow-y-auto">
              <LogDetailContent
                logEntry={enrichedLog}
                onOpenSettings={onOpenSettings}
                isLoadingDetails={isLoadingDetails}
                accessToken={accessToken ?? null}
              />
            </div>
          </div>
        </div>
    </Drawer>
  );
}

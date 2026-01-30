import { Button, Tag, Tooltip, Typography } from "antd";
import { CloseOutlined, CopyOutlined, UpOutlined, DownOutlined } from "@ant-design/icons";
import moment from "moment";
import { LogEntry } from "../columns";
import {
  DRAWER_HEADER_PADDING,
  COLOR_BORDER,
  COLOR_BACKGROUND,
  SPACING_MEDIUM,
  SPACING_LARGE,
  FONT_SIZE_HEADER,
  FONT_SIZE_MEDIUM,
  FONT_FAMILY_MONO,
  SPACING_SMALL,
} from "./constants";

const { Text } = Typography;

interface DrawerHeaderProps {
  log: LogEntry;
  onClose: () => void;
  onCopyRequestId: () => void;
  onPrevious: () => void;
  onNext: () => void;
  statusLabel: string;
  statusColor: "error" | "success";
  environment: string;
}

/**
 * Header component for the log details drawer.
 * Displays request ID, navigation controls, status, environment, and timestamp.
 */
export function DrawerHeader({
  log,
  onClose,
  onCopyRequestId,
  onPrevious,
  onNext,
  statusLabel,
  statusColor,
  environment,
}: DrawerHeaderProps) {
  return (
    <div
      style={{
        padding: DRAWER_HEADER_PADDING,
        borderBottom: `1px solid ${COLOR_BORDER}`,
        backgroundColor: COLOR_BACKGROUND,
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      {/* Row 1: Request ID + Actions */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: SPACING_MEDIUM }}>
        <RequestIdSection requestId={log.request_id} onCopy={onCopyRequestId} />
        <NavigationSection onPrevious={onPrevious} onNext={onNext} onClose={onClose} />
      </div>

      {/* Row 2: Status + Env + Timestamp */}
      <StatusBar log={log} statusLabel={statusLabel} statusColor={statusColor} environment={environment} />
    </div>
  );
}

/**
 * Request ID display with copy button
 */
function RequestIdSection({ requestId, onCopy }: { requestId: string; onCopy: () => void }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: SPACING_MEDIUM, flex: 1, minWidth: 0 }}>
      <Tooltip title={requestId}>
        <Text
          strong
          style={{
            fontSize: FONT_SIZE_HEADER,
            fontFamily: FONT_FAMILY_MONO,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {requestId}
        </Text>
      </Tooltip>
      <Tooltip title="Copy Request ID">
        <Button type="text" size="small" icon={<CopyOutlined />} onClick={onCopy} />
      </Tooltip>
    </div>
  );
}

/**
 * Navigation controls (previous, next, close)
 */
function NavigationSection({
  onPrevious,
  onNext,
  onClose,
}: {
  onPrevious: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: SPACING_SMALL }}>
      <Tooltip title="Previous (K)">
        <Button type="text" size="small" icon={<UpOutlined />} onClick={onPrevious} />
      </Tooltip>
      <Tooltip title="Next (J)">
        <Button type="text" size="small" icon={<DownOutlined />} onClick={onNext} />
      </Tooltip>

      <div style={{ width: 1, height: 20, background: COLOR_BORDER, margin: `0 ${SPACING_MEDIUM}px` }} />

      <Button type="text" icon={<CloseOutlined />} onClick={onClose} />
    </div>
  );
}

/**
 * Status bar with tags and timestamp
 */
function StatusBar({
  log,
  statusLabel,
  statusColor,
  environment,
}: {
  log: LogEntry;
  statusLabel: string;
  statusColor: "error" | "success";
  environment: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: SPACING_LARGE }}>
      <Tag color={statusColor}>{statusLabel}</Tag>
      <Tag>Env: {environment}</Tag>
      <Text type="secondary" style={{ fontSize: FONT_SIZE_MEDIUM }}>
        {moment(log.startTime).format("MMM D, YYYY h:mm:ss A")}
        <span style={{ marginLeft: SPACING_MEDIUM }}>({moment(log.startTime).fromNow()})</span>
      </Text>
    </div>
  );
}

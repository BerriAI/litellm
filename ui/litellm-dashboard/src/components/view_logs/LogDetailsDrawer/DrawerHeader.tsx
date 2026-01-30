import { Button, Tag, Tooltip, Typography } from "antd";
import { CloseOutlined, CopyOutlined, UpOutlined, DownOutlined } from "@ant-design/icons";
import moment from "moment";
import { LogEntry } from "../columns";
import { getProviderLogoAndName } from "../../provider_info_helpers";
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
 * Displays model/provider, request ID, navigation controls, status, environment, and timestamp.
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
  const provider = log.custom_llm_provider || "";
  const providerInfo = provider ? getProviderLogoAndName(provider) : null;

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
      {/* Row 0: Model + Provider with Logo */}
      <ModelProviderSection model={log.model} providerLogo={providerInfo?.logo} providerName={providerInfo?.displayName} />

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
 * Model and Provider display with logo
 */
function ModelProviderSection({
  model,
  providerLogo,
  providerName,
}: {
  model: string;
  providerLogo?: string;
  providerName?: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: SPACING_MEDIUM, marginBottom: SPACING_MEDIUM }}>
      {providerLogo && (
        <img
          src={providerLogo}
          alt={providerName || "Provider"}
          style={{ width: 24, height: 24 }}
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.style.display = "none";
          }}
        />
      )}
      <div>
        <Text strong style={{ fontSize: 14 }}>
          {model}
        </Text>
        {providerName && (
          <Text type="secondary" style={{ fontSize: 12, marginLeft: SPACING_MEDIUM }}>
            {providerName}
          </Text>
        )}
      </div>
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
 * Shows keyboard shortcuts with bounding boxes for visibility
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
  const keyboardShortcutStyle = {
    border: "1px solid #d9d9d9",
    borderRadius: 4,
    padding: "0 4px",
    fontSize: 12,
    fontFamily: "monospace",
    marginLeft: 4,
    background: "#fafafa",
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: SPACING_SMALL }}>
      <Button type="text" size="small" onClick={onPrevious}>
        <UpOutlined />
        <span style={keyboardShortcutStyle}>K</span>
      </Button>
      <Button type="text" size="small" onClick={onNext}>
        <DownOutlined />
        <span style={keyboardShortcutStyle}>J</span>
      </Button>

      <div style={{ width: 1, height: 20, background: COLOR_BORDER, margin: `0 ${SPACING_MEDIUM}px` }} />

      <Tooltip title="ESC to close">
        <Button type="text" icon={<CloseOutlined />} onClick={onClose} />
      </Tooltip>
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

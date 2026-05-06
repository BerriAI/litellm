"use client";

/**
 * SessionRow — single entry in the SessionList sidebar.
 *
 * Status pill colors map session.status to antd Tag colors. We avoid
 * "yellow" intentionally (per CLAUDE.md UI rules); use "gold" for amber.
 */
import Link from "next/link";
import { Tag, Typography } from "antd";
import { relativeOrAbsolute } from "@/app/(dashboard)/agents/components/_dayjs";
import type { CloudAgentSession, CloudAgentSessionStatus } from "@/types/cloud-agents";

const { Text, Paragraph } = Typography;

const STATUS_TO_COLOR: Record<CloudAgentSessionStatus, string> = {
  provisioning: "gold",
  running: "blue",
  paused: "default",
  completed: "green",
  failed: "red",
};

interface SessionRowProps {
  session: CloudAgentSession;
  active: boolean;
}

export default function SessionRow({ session, active }: SessionRowProps) {
  const updatedLabel = relativeOrAbsolute(session.updated_at);
  return (
    <Link
      href={`/agents/${session.agent_id}/sessions/${session.session_id}`}
      data-testid={`session-row-${session.session_id}`}
      data-active={active ? "true" : "false"}
      className={`block border-l-2 px-3 py-2 transition-colors ${
        active ? "border-blue-500 bg-blue-50" : "border-transparent hover:bg-gray-50"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <Text strong className="!truncate !text-sm">
          {session.title || "Untitled session"}
        </Text>
        <Tag color={STATUS_TO_COLOR[session.status]} className="!m-0 !text-xs">
          {session.status}
        </Tag>
      </div>
      <Paragraph type="secondary" className="!mb-0 !mt-1 !text-xs">
        {session.branch} • {updatedLabel}
      </Paragraph>
    </Link>
  );
}

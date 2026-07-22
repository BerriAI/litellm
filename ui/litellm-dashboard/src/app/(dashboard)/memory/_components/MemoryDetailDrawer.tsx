"use client";

import { Drawer, Space, Typography } from "antd";
import React from "react";

import { MemoryRow } from "@/components/networking";

const { Text, Paragraph } = Typography;

interface MemoryDetailDrawerProps {
  row: MemoryRow | null;
  onClose: () => void;
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

export function MemoryDetailDrawer({ row, onClose }: MemoryDetailDrawerProps) {
  return (
    <Drawer
      open={!!row}
      onClose={onClose}
      title={
        row ? (
          <Space>
            <Text code>{row.key}</Text>
          </Space>
        ) : (
          "Memory"
        )
      }
      width={720}
      destroyOnClose
    >
      {row && (
        <Space direction="vertical" size="middle" style={{ width: "100%" }}>
          <Space size="large" wrap>
            <div>
              <Text strong style={{ display: "block" }}>
                Memory ID
              </Text>
              <Text code style={{ fontSize: 12 }}>
                {row.memory_id}
              </Text>
            </div>
            <div>
              <Text strong style={{ display: "block" }}>
                User ID
              </Text>
              <Text type={row.user_id ? undefined : "secondary"}>{row.user_id ?? "-"}</Text>
            </div>
            <div>
              <Text strong style={{ display: "block" }}>
                Team ID
              </Text>
              <Text type={row.team_id ? undefined : "secondary"}>{row.team_id ?? "-"}</Text>
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
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                fontSize: 13,
              }}
            >
              {row.value}
            </Paragraph>
          </div>
          {row.metadata !== undefined && row.metadata !== null && (
            <div>
              <Text strong>Metadata</Text>
              <Paragraph
                style={{
                  background: "#fafafa",
                  padding: 12,
                  borderRadius: 6,
                  whiteSpace: "pre-wrap",
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                  fontSize: 12,
                }}
              >
                {JSON.stringify(row.metadata, null, 2)}
              </Paragraph>
            </div>
          )}
          <Space split={<Text type="secondary">·</Text>} wrap size="small" style={{ color: "rgba(0,0,0,0.45)" }}>
            <Text type="secondary">
              Created {formatTimestamp(row.created_at)}
              {row.created_by ? ` by ${row.created_by}` : ""}
            </Text>
            <Text type="secondary">
              Updated {formatTimestamp(row.updated_at)}
              {row.updated_by ? ` by ${row.updated_by}` : ""}
            </Text>
          </Space>
        </Space>
      )}
    </Drawer>
  );
}

export default MemoryDetailDrawer;

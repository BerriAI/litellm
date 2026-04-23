import React from "react";
import { Card, Tag, Table, Typography, Space, Tooltip } from "antd";
import { CheckCircleOutlined, CloseCircleOutlined, ExperimentOutlined } from "@ant-design/icons";

const { Text } = Typography;

interface EvalVerdict {
  criterion_name: string;
  score: number;
  reasoning: string;
  passed: boolean;
}

interface EvalInformation {
  eval_id?: string;
  eval_name: string;
  overall_score: number;
  passed: boolean;
  judge_model: string;
  iteration?: number;
  eval_error?: string | null;
  verdicts?: EvalVerdict[];
  threshold?: number;
}

interface EvalViewerProps {
  data: EvalInformation | EvalInformation[];
}

export default function EvalViewer({ data }: EvalViewerProps) {
  const entries: EvalInformation[] = Array.isArray(data) ? data : [data];

  if (!entries.length) return null;

  return (
    <div className="mb-6">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <ExperimentOutlined style={{ fontSize: 16, color: "#6366f1" }} />
        <Text strong style={{ fontSize: 15 }}>
          Evals
        </Text>
        <Tag color="blue" style={{ fontSize: 11 }}>
          Beta
        </Tag>
      </div>

      {entries.map((entry, idx) => (
        <EvalEntryCard key={entry.eval_id || idx} entry={entry} />
      ))}
    </div>
  );
}

function EvalEntryCard({ entry }: { entry: EvalInformation }) {
  const passed = entry.passed;
  const scoreColor = passed ? "#52c41a" : "#ff4d4f";

  const columns = [
    {
      title: "Criterion",
      dataIndex: "criterion_name",
      key: "criterion_name",
      render: (v: string) => <Text strong>{v}</Text>,
    },
    {
      title: "Score",
      dataIndex: "score",
      key: "score",
      width: 80,
      render: (v: number) => (
        <Text style={{ color: v >= 70 ? "#52c41a" : v >= 50 ? "#faad14" : "#ff4d4f", fontWeight: 600 }}>
          {v}
        </Text>
      ),
    },
    {
      title: "Comment",
      dataIndex: "reasoning",
      key: "reasoning",
      render: (v: string) => (
        <Tooltip title={v}>
          <Text style={{ fontSize: 12 }} ellipsis>
            {v}
          </Text>
        </Tooltip>
      ),
    },
  ];

  return (
    <Card
      size="small"
      className="mb-3"
      style={{ borderLeft: `3px solid ${scoreColor}` }}
      title={
        <Space>
          {passed ? (
            <CheckCircleOutlined style={{ color: "#52c41a" }} />
          ) : (
            <CloseCircleOutlined style={{ color: "#ff4d4f" }} />
          )}
          <Text strong>{entry.eval_name}</Text>
          <Tag color={passed ? "success" : "error"}>{passed ? "PASSED" : "FAILED"}</Tag>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {entry.overall_score?.toFixed(0)} / 100
            {entry.threshold != null && ` (threshold: ${entry.threshold})`}
          </Text>
        </Space>
      }
      extra={
        <Space size="small">
          {entry.judge_model && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              Judge: {entry.judge_model}
            </Text>
          )}
          {entry.iteration != null && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              Iter: {entry.iteration + 1}
            </Text>
          )}
        </Space>
      }
    >
      {entry.eval_error && (
        <Text type="warning" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
          Judge error: {entry.eval_error}
        </Text>
      )}

      {entry.verdicts && entry.verdicts.length > 0 ? (
        <Table
          dataSource={entry.verdicts}
          columns={columns}
          pagination={false}
          size="small"
          rowKey="criterion_name"
        />
      ) : (
        <Text type="secondary" style={{ fontSize: 12 }}>
          Score: {entry.overall_score?.toFixed(1)} — no per-criterion breakdown available.
        </Text>
      )}
    </Card>
  );
}

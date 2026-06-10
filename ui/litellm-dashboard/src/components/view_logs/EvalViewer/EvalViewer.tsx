import React from "react";
import { Card, Tag, Table, Typography, Space, Tooltip } from "antd";
import { CheckCircleOutlined, CloseCircleOutlined, ExperimentOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";

const { Text } = Typography;

interface EvalVerdict {
  criterion_name: string;
  score: number;
  reasoning: string;
  passed: boolean;
  weight?: number;
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
  const { t } = useTranslation();
  const entries: EvalInformation[] = Array.isArray(data) ? data : [data];

  if (!entries.length) return null;

  return (
    <div className="mb-6">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <ExperimentOutlined style={{ fontSize: 16, color: "#6366f1" }} />
        <Text strong style={{ fontSize: 15 }}>
          {t("viewLogs.evalViewer.title")}
        </Text>
      </div>

      {entries.map((entry, idx) => (
        <EvalEntryCard key={entry.eval_id || idx} entry={entry} />
      ))}
    </div>
  );
}

function EvalEntryCard({ entry }: { entry: EvalInformation }) {
  const { t } = useTranslation();
  const passed = entry.passed;
  const scoreColor = passed ? "#52c41a" : "#ff4d4f";

  // Filter out synthetic "Overall" row the judge sometimes appends — it's already in the header
  const verdicts = (entry.verdicts || []).filter((v) => (v.criterion_name || "").toLowerCase() !== "overall");

  const columns = [
    {
      title: t("viewLogs.evalViewer.colCriterion"),
      dataIndex: "criterion_name",
      key: "criterion_name",
      width: 160,
      render: (v: string) => (
        <Text strong style={{ whiteSpace: "nowrap" }}>
          {v}
        </Text>
      ),
    },
    {
      title: t("viewLogs.evalViewer.colWeight"),
      dataIndex: "weight",
      key: "weight",
      width: 65,
      render: (v: number) =>
        v != null ? (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {v}%
          </Text>
        ) : null,
    },
    {
      title: t("viewLogs.evalViewer.colScore"),
      dataIndex: "score",
      key: "score",
      width: 65,
      render: (v: number) => (
        <Text style={{ color: v >= 70 ? "#52c41a" : v >= 50 ? "#faad14" : "#ff4d4f", fontWeight: 600 }}>{v}</Text>
      ),
    },
    {
      title: (
        <Tooltip title={t("viewLogs.evalViewer.weightedTooltip")}>
          <span style={{ borderBottom: "1px dashed #aaa", cursor: "help" }}>
            {t("viewLogs.evalViewer.colWeighted")}
          </span>
        </Tooltip>
      ),
      key: "weighted",
      width: 75,
      render: (_: unknown, row: EvalVerdict) => {
        if (row.weight == null) return null;
        const contrib = (row.score * row.weight) / 100;
        return (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {contrib % 1 === 0 ? contrib : contrib.toFixed(1)}
          </Text>
        );
      },
    },
    {
      title: t("viewLogs.evalViewer.colComment"),
      dataIndex: "reasoning",
      key: "reasoning",
      ellipsis: { showTitle: false },
      render: (v: string) => (
        <Tooltip title={v}>
          <span style={{ fontSize: 12 }}>{v}</span>
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
          <Tag color={passed ? "success" : "error"}>
            {passed ? t("viewLogs.evalViewer.passed") : t("viewLogs.evalViewer.failed")}
          </Tag>
          <Tooltip title={t("viewLogs.evalViewer.overallScoreTooltip")}>
            <Text type="secondary" style={{ fontSize: 12, cursor: "help", borderBottom: "1px dashed #aaa" }}>
              {entry.overall_score?.toFixed(0)} / 100
              {entry.threshold != null && ` (${t("viewLogs.evalViewer.threshold")}: ${entry.threshold})`}
            </Text>
          </Tooltip>
        </Space>
      }
      extra={
        <Space size="small">
          {entry.judge_model && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              {t("viewLogs.evalViewer.judge")}: {entry.judge_model}
            </Text>
          )}
          {entry.iteration != null && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              {t("viewLogs.evalViewer.iter")}: {entry.iteration + 1}
            </Text>
          )}
        </Space>
      }
    >
      {entry.eval_error && (
        <Text type="warning" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
          {t("viewLogs.evalViewer.judgeError")}: {entry.eval_error}
        </Text>
      )}

      {verdicts.length > 0 ? (
        <Table
          dataSource={verdicts}
          columns={columns}
          pagination={false}
          size="small"
          rowKey="criterion_name"
          scroll={{ x: true }}
          summary={() => {
            const hasWeights = verdicts.some((v) => v.weight != null);
            if (!hasWeights) return null;
            const total = verdicts.reduce((sum, v) => sum + (v.weight != null ? (v.score * v.weight) / 100 : 0), 0);
            return (
              <Table.Summary.Row>
                <Table.Summary.Cell index={0}>
                  <Text strong style={{ fontSize: 12 }}>
                    {t("common.total")}
                  </Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={1} />
                <Table.Summary.Cell index={2} />
                <Table.Summary.Cell index={3}>
                  <Text strong style={{ fontSize: 12, color: scoreColor }}>
                    {total % 1 === 0 ? total : total.toFixed(1)}
                  </Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={4} />
              </Table.Summary.Row>
            );
          }}
        />
      ) : (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {t("viewLogs.evalViewer.scoreNoCriterion", { score: entry.overall_score?.toFixed(1) })}
        </Text>
      )}
    </Card>
  );
}

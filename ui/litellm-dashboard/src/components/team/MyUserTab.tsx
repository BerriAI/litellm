import { formatBudgetReset } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Card, Col, Progress, Row, Space, Tag, Tooltip, Typography } from "antd";
import React from "react";
import { getBudgetDurationLabel } from "../common_components/budget_duration_dropdown";
import { ModelMaxBudgetUsageOverview } from "../key_team_helpers/ModelMaxBudgetUsageOverview";
import { useMyTeamMember } from "./useMyTeamMember";

interface MyUserTabProps {
  teamId: string;
}

const labelWithTooltip = (label: string, tooltip: string) => (
  <Space size={4}>
    <Typography.Text type="secondary">{label}</Typography.Text>
    <Tooltip title={tooltip}>
      <InfoCircleOutlined style={{ color: "#8c8c8c" }} />
    </Tooltip>
  </Space>
);

const formatNumber = (value: number | null | undefined, digits = 4): string => {
  if (value === null || value === undefined) return "0";
  return formatNumberWithCommas(value, digits);
};

const formatRateLimit = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "Unlimited";
  return formatNumberWithCommas(value, 0);
};

function budgetProgressPercent(spend: number, maxBudget: number | null): number | null {
  if (maxBudget === null || maxBudget <= 0) return null;
  return Math.min(Math.round((spend / maxBudget) * 1000) / 10, 999.9);
}

function progressStatus(percent: number): "success" | "normal" | "exception" {
  if (percent >= 100) return "exception";
  if (percent >= 80) return "normal";
  return "success";
}

function progressStrokeColor(percent: number): string {
  if (percent >= 100) return "#ff4d4f";
  if (percent >= 80) return "#faad14";
  return "#52c41a";
}

export default function MyUserTab({ teamId }: MyUserTabProps) {
  const { data, isLoading, error } = useMyTeamMember(teamId);

  if (isLoading) {
    return (
      <Card>
        <Typography.Text type="secondary">Loading your membership info…</Typography.Text>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Typography.Text type="danger">
          {error instanceof Error ? error.message : "Failed to load your membership info for this team."}
        </Typography.Text>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <Typography.Text type="secondary">
          You are not a member of this team, so there is no personal budget to show. Ask a team admin
          to add you on the Members tab (role: user), then refresh this page.
        </Typography.Text>
      </Card>
    );
  }

  const budgetTable = data.litellm_budget_table ?? null;
  const maxBudget = budgetTable?.max_budget ?? null;
  const spend = data.spend ?? 0;
  const totalSpend = data.total_spend ?? 0;
  const tpmLimit = budgetTable?.tpm_limit ?? null;
  const rpmLimit = budgetTable?.rpm_limit ?? null;
  const budgetReset = formatBudgetReset(budgetTable?.budget_reset_at);
  const allowedModels = budgetTable?.allowed_models ?? null;
  const percentUsed = budgetProgressPercent(spend, maxBudget);
  const remaining = maxBudget !== null ? Math.max(maxBudget - spend, 0) : null;
  const durationLabel = getBudgetDurationLabel(budgetTable?.budget_duration);
  const hasModelUsage =
    data.model_max_budget_usage != null && Object.keys(data.model_max_budget_usage).length > 0;

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Card>
        <Row gutter={[24, 16]}>
          <Col xs={24} sm={12} md={8}>
            <Typography.Text type="secondary">User</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Typography.Text strong>{data.user_email || data.user_id}</Typography.Text>
            </div>
            <Typography.Text type="secondary" style={{ fontSize: 12, fontFamily: "monospace" }}>
              {data.user_id}
            </Typography.Text>
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Typography.Text type="secondary">Team Role</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Tag color={data.role === "admin" ? "blue" : "default"}>{data.role || "user"}</Tag>
            </div>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(
              "Your Budget This Cycle",
              "Spend across all your virtual keys on this team for the current budget window. This is what enforcement checks.",
            )}
            <div style={{ marginTop: 8 }}>
              <Typography.Title level={3} style={{ margin: 0 }}>
                ${formatNumber(spend, 4)}
              </Typography.Title>
              <Typography.Text type="secondary">
                of {maxBudget === null ? "Unlimited" : `$${formatNumber(maxBudget, 2)}`}
                {maxBudget !== null && budgetTable?.budget_duration ? ` (${durationLabel})` : ""}
              </Typography.Text>
              {data.using_team_default_budget && (
                <div style={{ marginTop: 4 }}>
                  <Tag>Team default</Tag>
                </div>
              )}
            </div>
            {percentUsed !== null && (
              <div style={{ marginTop: 12 }}>
                <Progress
                  percent={Math.min(percentUsed, 100)}
                  status={progressStatus(percentUsed)}
                  strokeColor={progressStrokeColor(percentUsed)}
                  format={() => `${percentUsed}%`}
                />
                <Typography.Text type="secondary">
                  ${formatNumber(remaining, 2)} remaining
                </Typography.Text>
              </div>
            )}
            {budgetReset && (
              <div style={{ marginTop: 4 }}>
                <Typography.Text type="secondary">Resets {budgetReset}</Typography.Text>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip("Rate Limits", "Your per-member rate limits within this team.")}
            <div style={{ marginTop: 8 }}>
              <Typography.Text>TPM: {formatRateLimit(tpmLimit)}</Typography.Text>
              <br />
              <Typography.Text>RPM: {formatRateLimit(rpmLimit)}</Typography.Text>
            </div>
          </Card>
        </Col>

        <Col xs={24}>
          <Card>
            {labelWithTooltip(
              "Per-Model Budgets",
              "Model-specific caps still count toward your overall per-user budget. Spend is shared across your keys on this team.",
            )}
            <div style={{ marginTop: 12 }}>
              {hasModelUsage ? (
                <ModelMaxBudgetUsageOverview usage={data.model_max_budget_usage} />
              ) : (
                <Typography.Text type="secondary">No per-model budgets configured for you</Typography.Text>
              )}
            </div>
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip("Total Spend (USD)", "Cumulative spend across all budget cycles within this team.")}
            <div style={{ marginTop: 8 }}>
              <Typography.Title level={4} style={{ margin: 0 }}>
                ${formatNumber(totalSpend, 4)}
              </Typography.Title>
            </div>
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip("Model Scope", "Models you can access within this team.")}
            <div style={{ marginTop: 8 }}>
              {allowedModels && allowedModels.length > 0 ? (
                <Space wrap>
                  {allowedModels.map((m) => (
                    <Tag key={m}>{m}</Tag>
                  ))}
                </Space>
              ) : (
                <Typography.Text>All Team Models</Typography.Text>
              )}
            </div>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}

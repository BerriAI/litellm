import { formatBudgetReset } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Card, Col, Row, Space, Tag, Tooltip, Typography } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";
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

const formatRateLimit = (value: number | null | undefined, unlimitedLabel: string): string => {
  if (value === null || value === undefined) return unlimitedLabel;
  return formatNumberWithCommas(value, 0);
};

export default function MyUserTab({ teamId }: MyUserTabProps) {
  const { t } = useTranslation();
  const { data, isLoading, error } = useMyTeamMember(teamId);

  if (isLoading) {
    return (
      <Card>
        <Typography.Text type="secondary">{t("teamPage.myUserTab.loadingText")}</Typography.Text>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Typography.Text type="danger">
          {error instanceof Error ? error.message : t("teamPage.myUserTab.errorFallback")}
        </Typography.Text>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <Typography.Text type="secondary">{t("teamPage.myUserTab.noDataText")}</Typography.Text>
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

  const unlimitedLabel = t("teamPage.myUserTab.unlimited");

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Card>
        <Row gutter={[24, 16]}>
          <Col xs={24} sm={12} md={8}>
            <Typography.Text type="secondary">{t("teamPage.myUserTab.userLabel")}</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Typography.Text strong>{data.user_email || data.user_id}</Typography.Text>
            </div>
            <Typography.Text type="secondary" style={{ fontSize: 12, fontFamily: "monospace" }}>
              {data.user_id}
            </Typography.Text>
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Typography.Text type="secondary">{t("teamPage.myUserTab.teamRoleLabel")}</Typography.Text>
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
              t("teamPage.myUserTab.currentCycleSpendLabel"),
              t("teamPage.myUserTab.currentCycleSpendTooltip"),
            )}
            <div style={{ marginTop: 8 }}>
              <Typography.Title level={3} style={{ margin: 0 }}>
                ${formatNumber(spend, 4)}
              </Typography.Title>
              <Typography.Text type="secondary">
                {t("teamPage.myUserTab.ofBudget", {
                  budget: maxBudget === null ? unlimitedLabel : `$${formatNumber(maxBudget, 4)}`,
                })}
              </Typography.Text>
            </div>
            {budgetReset && (
              <div style={{ marginTop: 4 }}>
                <Typography.Text type="secondary">
                  {t("teamPage.myUserTab.resetsAt", { when: budgetReset })}
                </Typography.Text>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(t("teamPage.myUserTab.rateLimitsLabel"), t("teamPage.myUserTab.rateLimitsTooltip"))}
            <div style={{ marginTop: 8 }}>
              <Typography.Text>
                {t("teamPage.myUserTab.tpmValue", { value: formatRateLimit(tpmLimit, unlimitedLabel) })}
              </Typography.Text>
              <br />
              <Typography.Text>
                {t("teamPage.myUserTab.rpmValue", { value: formatRateLimit(rpmLimit, unlimitedLabel) })}
              </Typography.Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(t("teamPage.myUserTab.totalSpendLabel"), t("teamPage.myUserTab.totalSpendTooltip"))}
            <div style={{ marginTop: 8 }}>
              <Typography.Title level={4} style={{ margin: 0 }}>
                ${formatNumber(totalSpend, 4)}
              </Typography.Title>
            </div>
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(t("teamPage.myUserTab.modelScopeLabel"), t("teamPage.myUserTab.modelScopeTooltip"))}
            <div style={{ marginTop: 8 }}>
              {allowedModels && allowedModels.length > 0 ? (
                <Space wrap>
                  {allowedModels.map((m) => (
                    <Tag key={m}>{m}</Tag>
                  ))}
                </Space>
              ) : (
                <Typography.Text>{t("teamPage.myUserTab.allTeamModels")}</Typography.Text>
              )}
            </div>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}

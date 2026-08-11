import { formatNumberWithCommas } from "@/utils/dataUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Card, Col, Row, Space, Tag, Tooltip, Typography } from "antd";
import React from "react";
import { useMyTeamMember } from "./useMyTeamMember";
import { useTranslation } from "react-i18next";

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

const formatRateLimit = (value: number | null | undefined, unlimited: string): string => {
  if (value === null || value === undefined) return unlimited;
  return formatNumberWithCommas(value, 0);
};

export default function MyUserTab({ teamId }: MyUserTabProps) {
  const { t, i18n } = useTranslation("gateway");
  const { data, isLoading, error } = useMyTeamMember(teamId);

  if (isLoading) {
    return (
      <Card>
        <Typography.Text type="secondary">{t("teams.myMembership.loading")}</Typography.Text>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Typography.Text type="danger">
          {error instanceof Error ? error.message : t("teams.myMembership.loadFailed")}
        </Typography.Text>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <Typography.Text type="secondary">{t("teams.myMembership.empty")}</Typography.Text>
      </Card>
    );
  }

  const budgetTable = data.litellm_budget_table ?? null;
  const maxBudget = budgetTable?.max_budget ?? null;
  const spend = data.spend ?? 0;
  const totalSpend = data.total_spend ?? 0;
  const tpmLimit = budgetTable?.tpm_limit ?? null;
  const rpmLimit = budgetTable?.rpm_limit ?? null;
  const budgetReset = budgetTable?.budget_reset_at
    ? new Intl.DateTimeFormat(i18n.resolvedLanguage === "ru" ? "ru-RU" : "en-US", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }).format(new Date(budgetTable.budget_reset_at))
    : null;
  const allowedModels = budgetTable?.allowed_models ?? null;

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Card>
        <Row gutter={[24, 16]}>
          <Col xs={24} sm={12} md={8}>
            <Typography.Text type="secondary">{t("teams.myMembership.user")}</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Typography.Text strong>{data.user_email || data.user_id}</Typography.Text>
            </div>
            <Typography.Text type="secondary" style={{ fontSize: 12, fontFamily: "monospace" }}>
              {data.user_id}
            </Typography.Text>
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Typography.Text type="secondary">{t("teams.myMembership.teamRole")}</Typography.Text>
            <div style={{ marginTop: 4 }}>
              <Tag color={data.role === "admin" ? "blue" : "default"}>{data.role || "user"}</Tag>
            </div>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(t("teams.myMembership.currentSpend"), t("teams.myMembership.currentSpendTooltip"))}
            <div style={{ marginTop: 8 }}>
              <Typography.Title level={3} style={{ margin: 0 }}>
                ${formatNumber(spend, 4)}
              </Typography.Title>
              <Typography.Text type="secondary">
                {t("teams.table.of")}{" "}
                {maxBudget === null ? t("teams.table.unlimited") : `$${formatNumber(maxBudget, 4)}`}
              </Typography.Text>
            </div>
            {budgetReset && (
              <div style={{ marginTop: 4 }}>
                <Typography.Text type="secondary">
                  {t("teams.myMembership.resets", { date: budgetReset })}
                </Typography.Text>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(t("teams.table.rateLimits"), t("teams.myMembership.rateLimitsTooltip"))}
            <div style={{ marginTop: 8 }}>
              <Typography.Text>TPM: {formatRateLimit(tpmLimit, t("teams.table.unlimited"))}</Typography.Text>
              <br />
              <Typography.Text>RPM: {formatRateLimit(rpmLimit, t("teams.table.unlimited"))}</Typography.Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(t("teams.myMembership.totalSpend"), t("teams.myMembership.totalSpendTooltip"))}
            <div style={{ marginTop: 8 }}>
              <Typography.Title level={4} style={{ margin: 0 }}>
                ${formatNumber(totalSpend, 4)}
              </Typography.Title>
            </div>
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card>
            {labelWithTooltip(t("teams.myMembership.modelScope"), t("teams.myMembership.modelScopeTooltip"))}
            <div style={{ marginTop: 8 }}>
              {allowedModels && allowedModels.length > 0 ? (
                <Space wrap>
                  {allowedModels.map((m) => (
                    <Tag key={m}>{m}</Tag>
                  ))}
                </Space>
              ) : (
                <Typography.Text>{t("teams.myMembership.allTeamModels")}</Typography.Text>
              )}
            </div>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}

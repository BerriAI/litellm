import React from "react";
import { Button, Typography, Tooltip, Space, Divider, Flex, Popover } from "antd";
import {
  ArrowLeftOutlined,
  SyncOutlined,
  DeleteOutlined,
  PlusOutlined,
  UserOutlined,
  CalendarOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  SafetyCertificateOutlined,
  TransactionOutlined,
  FieldTimeOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import LabeledField from "../common_components/LabeledField";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";

const { Title, Text } = Typography;

export interface KeyInfoData {
  keyName: string;
  keyId: string;
  userId: string;
  userEmail: string;
  userAlias?: string | null;
  createdBy: string;
  createdAt: string;
  lastUpdated: string;
  lastActive: string;
  expires: string;
}

interface KeyInfoHeaderProps {
  data: KeyInfoData;
  onBack?: () => void;
  onCreateNew?: () => void;
  onRegenerate?: () => void;
  onDelete?: () => void;
  onResetSpend?: () => void;
  canModifyKey?: boolean;
  backButtonText?: string;
  regenerateDisabled?: boolean;
  regenerateTooltip?: string;
}

function UserField({ userAlias, userEmail, userId }: { userAlias?: string | null; userEmail: string; userId: string }) {
  const { t } = useTranslation();
  const labelEl = (
    <Space size={4}>
      <Text type="secondary">
        <UserOutlined />
      </Text>
      <Text type="secondary" style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {t("templates.keyInfoHeader.user")}
      </Text>
    </Space>
  );

  const isEmpty = !userAlias && !userEmail && !userId;
  if (isEmpty) {
    return (
      <div>
        {labelEl}
        <div>
          <Text strong>-</Text>
        </div>
      </div>
    );
  }

  const isDefaultAdmin = userId === "default_user_id";
  const displayValue = userAlias || userEmail || userId;

  const popoverContent = (
    <div className="flex flex-col gap-2 text-xs min-w-[200px] max-w-[300px]">
      {[
        { label: t("templates.keyInfoHeader.userAlias"), value: userAlias ?? null },
        { label: t("templates.keyInfoHeader.userEmail"), value: userEmail || null },
        { label: t("templates.keyInfoHeader.userId"), value: userId || null },
      ].map(({ label, value }) => (
        <div key={label} className="flex flex-col min-w-0">
          <span className="text-gray-400">{label}</span>
          {value ? (
            <Typography.Text
              className="font-mono text-xs"
              style={{ maxWidth: 220 }}
              ellipsis={{ tooltip: value }}
              copyable
            >
              {value}
            </Typography.Text>
          ) : (
            <span className="font-mono">-</span>
          )}
        </div>
      ))}
    </div>
  );

  if (isDefaultAdmin && !userAlias && !userEmail) {
    return (
      <div>
        {labelEl}
        <div>
          <Popover content={popoverContent} trigger="hover" placement="bottomLeft">
            <span className="cursor-default">
              <DefaultProxyAdminTag userId={userId} />
            </span>
          </Popover>
        </div>
      </div>
    );
  }

  return (
    <div>
      {labelEl}
      <div>
        <Popover content={popoverContent} trigger="hover" placement="bottomLeft">
          <Text strong ellipsis style={{ cursor: "default", maxWidth: 200, display: "block" }}>
            {displayValue}
          </Text>
        </Popover>
      </div>
    </div>
  );
}

export function KeyInfoHeader({
  data,
  onBack,
  onCreateNew,
  onRegenerate,
  onDelete,
  onResetSpend,
  canModifyKey = true,
  backButtonText,
  regenerateDisabled = false,
  regenerateTooltip,
}: KeyInfoHeaderProps) {
  const { t } = useTranslation();
  const resolvedBackButtonText = backButtonText ?? t("templates.keyInfoHeader.backToKeys");
  return (
    <div>
      {onCreateNew && (
        <div style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={onCreateNew}>
            {t("templates.keyInfoHeader.createNewKey")}
          </Button>
        </div>
      )}

      <div style={{ marginBottom: 16 }}>
        <Button type="text" icon={<ArrowLeftOutlined />} onClick={onBack}>
          {resolvedBackButtonText}
        </Button>
      </div>

      <Flex justify="space-between" align="start" style={{ marginBottom: 20 }}>
        <div>
          <Title
            level={3}
            copyable={{ tooltips: [t("templates.keyInfoHeader.copyKeyAlias"), t("common.copied")] }}
            style={{ margin: 0 }}
          >
            {data.keyName}
          </Title>
          <Text
            type="secondary"
            copyable={{ text: data.keyId, tooltips: [t("templates.keyInfoHeader.copyKeyId"), t("common.copied")] }}
          >
            {t("templates.keyInfoHeader.keyId", { keyId: data.keyId })}
          </Text>
        </div>
        {canModifyKey && (
          <Space>
            <Tooltip title={regenerateTooltip || ""}>
              <span>
                <Button icon={<SyncOutlined />} onClick={onRegenerate} disabled={regenerateDisabled}>
                  {t("templates.keyInfoHeader.regenerateKey")}
                </Button>
              </span>
            </Tooltip>
            {onResetSpend && (
              <Button danger icon={<TransactionOutlined />} onClick={onResetSpend}>
                {t("templates.keyInfoHeader.resetSpend")}
              </Button>
            )}
            <Button danger icon={<DeleteOutlined />} onClick={onDelete}>
              {t("templates.keyInfoHeader.deleteKey")}
            </Button>
          </Space>
        )}
      </Flex>

      <Flex align="stretch" gap={40} style={{ marginBottom: 40 }}>
        <Space direction="vertical" size={16}>
          <UserField userAlias={data.userAlias} userEmail={data.userEmail} userId={data.userId} />
          <LabeledField
            label={t("templates.keyInfoHeader.expires")}
            value={data.expires}
            icon={<FieldTimeOutlined />}
          />
        </Space>

        <Divider type="vertical" style={{ height: "auto" }} />

        <Space direction="vertical" size={16}>
          <LabeledField label={t("common.createdAt")} value={data.createdAt} icon={<CalendarOutlined />} />
          <LabeledField
            label={t("templates.keyInfoHeader.createdBy")}
            value={data.createdBy}
            icon={<SafetyCertificateOutlined />}
            truncate
            copyable
            defaultUserIdCheck
          />
        </Space>

        <Divider type="vertical" style={{ height: "auto" }} />

        <Space direction="vertical" size={16}>
          <LabeledField
            label={t("templates.keyInfoHeader.lastUpdated")}
            value={data.lastUpdated}
            icon={<ClockCircleOutlined />}
          />
          <LabeledField
            label={t("templates.keyInfoHeader.lastActive")}
            value={data.lastActive}
            icon={<ThunderboltOutlined />}
          />
        </Space>
      </Flex>
    </div>
  );
}

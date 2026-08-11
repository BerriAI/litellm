import React from "react";
import { Button, Typography, Tooltip, Space, Divider, Flex, Popover, Dropdown, Tag } from "antd";
import type { MenuProps } from "antd";
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
  MoreOutlined,
  StopOutlined,
  CheckCircleOutlined,
} from "@ant-design/icons";
import LabeledField from "../common_components/LabeledField";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import { useTranslation } from "react-i18next";

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
  onToggleBlocked?: () => void;
  isBlocked?: boolean;
  canModifyKey?: boolean;
  backButtonText?: string;
  regenerateDisabled?: boolean;
  regenerateTooltip?: string;
}

function UserField({ userAlias, userEmail, userId }: { userAlias?: string | null; userEmail: string; userId: string }) {
  const { t } = useTranslation("gateway");
  const labelEl = (
    <Space size={4}>
      <Text type="secondary">
        <UserOutlined />
      </Text>
      <Text type="secondary" style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {t("virtualKeys.columns.user")}
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
        { label: t("virtualKeys.columns.userAlias"), value: userAlias ?? null },
        { label: t("virtualKeys.columns.userEmail"), value: userEmail || null },
        { label: t("virtualKeys.columns.userId"), value: userId || null },
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
  onToggleBlocked,
  isBlocked = false,
  canModifyKey = true,
  backButtonText,
  regenerateDisabled = false,
  regenerateTooltip,
}: KeyInfoHeaderProps) {
  const { t } = useTranslation("gateway");
  const destructiveActionItems: MenuProps["items"] = [
    ...(onToggleBlocked
      ? [
          isBlocked
            ? { key: "unblock", label: t("virtualKeys.details.unblockKey"), icon: <CheckCircleOutlined /> }
            : { key: "block", label: t("virtualKeys.details.blockKey"), icon: <StopOutlined />, danger: true },
        ]
      : []),
    ...(onResetSpend
      ? [
          {
            key: "reset-spend",
            label: t("virtualKeys.details.resetSpend"),
            icon: <TransactionOutlined />,
            danger: true,
          },
        ]
      : []),
    { key: "delete", label: t("virtualKeys.details.deleteKey"), icon: <DeleteOutlined />, danger: true },
  ];

  const handleDestructiveActionClick: MenuProps["onClick"] = ({ key }) => {
    if (key === "block" || key === "unblock") onToggleBlocked?.();
    if (key === "reset-spend") onResetSpend?.();
    if (key === "delete") onDelete?.();
  };

  return (
    <div>
      {onCreateNew && (
        <div style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={onCreateNew}>
            {t("virtualKeys.details.createNewKey")}
          </Button>
        </div>
      )}

      <div style={{ marginBottom: 16 }}>
        <Button type="text" icon={<ArrowLeftOutlined />} onClick={onBack}>
          {backButtonText ?? t("virtualKeys.details.back")}
        </Button>
      </div>

      <Flex justify="space-between" align="start" style={{ marginBottom: 20 }}>
        <div>
          <Space align="center">
            <Title
              level={3}
              copyable={{ tooltips: [t("virtualKeys.details.copyAlias"), t("virtualKeys.details.copied")] }}
              style={{ margin: 0 }}
            >
              {data.keyName}
            </Title>
            {isBlocked && (
              <Tag color="red" icon={<StopOutlined />}>
                {t("virtualKeys.status.blocked")}
              </Tag>
            )}
          </Space>
          <Text
            type="secondary"
            copyable={{
              text: data.keyId,
              tooltips: [t("virtualKeys.details.copyId"), t("virtualKeys.details.copied")],
            }}
          >
            {t("virtualKeys.columns.keyId")}: {data.keyId}
          </Text>
        </div>
        {canModifyKey && (
          <Space>
            <Tooltip title={regenerateTooltip || ""}>
              <span>
                <Button icon={<SyncOutlined />} onClick={onRegenerate} disabled={regenerateDisabled}>
                  {t("virtualKeys.details.regenerateKey")}
                </Button>
              </span>
            </Tooltip>
            <Dropdown
              menu={{ items: destructiveActionItems, onClick: handleDestructiveActionClick }}
              trigger={["click"]}
            >
              <Button icon={<MoreOutlined />} aria-label={t("virtualKeys.details.moreActions")} />
            </Dropdown>
          </Space>
        )}
      </Flex>

      <Flex align="stretch" gap={40} style={{ marginBottom: 40 }}>
        <Space direction="vertical" size={16}>
          <UserField userAlias={data.userAlias} userEmail={data.userEmail} userId={data.userId} />
          <LabeledField label={t("virtualKeys.columns.expires")} value={data.expires} icon={<FieldTimeOutlined />} />
        </Space>

        <Divider type="vertical" style={{ height: "auto" }} />

        <Space direction="vertical" size={16}>
          <LabeledField label={t("virtualKeys.columns.createdAt")} value={data.createdAt} icon={<CalendarOutlined />} />
          <LabeledField
            label={t("virtualKeys.columns.createdBy")}
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
            label={t("virtualKeys.columns.updatedAt")}
            value={data.lastUpdated}
            icon={<ClockCircleOutlined />}
          />
          <LabeledField
            label={t("virtualKeys.columns.lastActive")}
            value={data.lastActive}
            icon={<ThunderboltOutlined />}
          />
        </Space>
      </Flex>
    </div>
  );
}

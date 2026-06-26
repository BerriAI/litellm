import {
  BankOutlined,
  BarChartOutlined,
  GlobalOutlined,
  LineChartOutlined,
  RobotOutlined,
  ShoppingCartOutlined,
  TagsOutlined,
  TeamOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Badge, Select } from "antd";
import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
export type UsageOption =
  | "global"
  | "my-usage"
  | "organization"
  | "team"
  | "customer"
  | "tag"
  | "agent"
  | "user"
  | "user-agent-activity";
export interface UsageViewSelectProps {
  value: UsageOption;
  onChange: (value: UsageOption) => void;
  isAdmin: boolean;
  canViewTagUsage?: boolean;
  title?: string;
  description?: string;
  "data-id"?: string;
}
interface OptionConfig {
  value: UsageOption;
  label: string;
  description: string;
  icon: React.ReactNode;
  adminOnly?: boolean;
  showForAdmin?: string;
  showForNonAdmin?: string;
  descriptionForAdmin?: string;
  descriptionForNonAdmin?: string;
  badgeText?: string;
}
const getOptions = (t: TFunction): OptionConfig[] => [
  {
    value: "global",
    label: t("usagePage.usageViewSelect.globalUsageLabel"),
    showForAdmin: t("usagePage.usageViewSelect.globalUsageLabel"),
    showForNonAdmin: t("usagePage.usageViewSelect.yourUsageLabel"),
    description: t("usagePage.usageViewSelect.globalUsageDesc"),
    descriptionForAdmin: t("usagePage.usageViewSelect.globalUsageDesc"),
    descriptionForNonAdmin: t("usagePage.usageViewSelect.yourUsageDesc"),
    icon: <GlobalOutlined style={{ fontSize: "16px" }} />,
  },
  {
    value: "my-usage",
    label: t("usagePage.usageViewSelect.yourUsageLabel"),
    description: t("usagePage.usageViewSelect.myUsageDesc"),
    icon: <UserOutlined style={{ fontSize: "16px" }} />,
    adminOnly: true,
  },
  {
    value: "organization",
    label: t("usagePage.usageViewSelect.organizationUsageLabel"),
    showForAdmin: t("usagePage.usageViewSelect.organizationUsageLabel"),
    showForNonAdmin: t("usagePage.usageViewSelect.yourOrganizationUsageLabel"),
    description: t("usagePage.usageViewSelect.organizationUsageDesc"),
    descriptionForAdmin: t("usagePage.usageViewSelect.organizationUsageAdminDesc"),
    descriptionForNonAdmin: t("usagePage.usageViewSelect.organizationUsageNonAdminDesc"),
    icon: <BankOutlined style={{ fontSize: "16px" }} />,
  },
  {
    value: "team",
    label: t("usagePage.usageViewSelect.teamUsageLabel"),
    description: t("usagePage.usageViewSelect.teamUsageDesc"),
    icon: <TeamOutlined style={{ fontSize: "16px" }} />,
  },
  {
    value: "customer",
    label: t("usagePage.usageViewSelect.customerUsageLabel"),
    description: t("usagePage.usageViewSelect.customerUsageDesc"),
    icon: <ShoppingCartOutlined style={{ fontSize: "16px" }} />,
    adminOnly: true,
  },
  {
    value: "tag",
    label: t("usagePage.usageViewSelect.tagUsageLabel"),
    description: t("usagePage.usageViewSelect.tagUsageDesc"),
    icon: <TagsOutlined style={{ fontSize: "16px" }} />,
    adminOnly: true,
  },
  {
    value: "agent",
    label: t("usagePage.usageViewSelect.agentUsageLabel"),
    description: t("usagePage.usageViewSelect.agentUsageDesc"),
    icon: <RobotOutlined style={{ fontSize: "16px" }} />,
    adminOnly: true,
  },
  {
    value: "user",
    label: t("usagePage.usageViewSelect.userUsageLabel"),
    description: t("usagePage.usageViewSelect.userUsageDesc"),
    icon: <UserOutlined style={{ fontSize: "16px" }} />,
    adminOnly: true,
  },
  {
    value: "user-agent-activity",
    label: t("usagePage.usageViewSelect.userAgentActivityLabel"),
    description: t("usagePage.usageViewSelect.userAgentActivityDesc"),
    icon: <LineChartOutlined style={{ fontSize: "16px" }} />,
    adminOnly: true,
  },
];
export const UsageViewSelect: React.FC<UsageViewSelectProps> = ({
  value,
  onChange,
  isAdmin,
  canViewTagUsage = false,
  title,
  description,
  "data-id": dataId,
}) => {
  const { t } = useTranslation();
  const options = useMemo(() => getOptions(t), [t]);
  const resolvedTitle = title ?? t("usagePage.usageViewSelect.title");
  const resolvedDescription = description ?? t("usagePage.usageViewSelect.description");
  const getFilteredOptions = () => {
    return options
      .filter((option) => {
        if (option.value === "tag" && canViewTagUsage) {
          return true;
        }
        if (option.adminOnly && !isAdmin) {
          return false;
        }
        return true;
      })
      .map((option) => {
        let label = option.label;
        let desc = option.description;
        if (option.showForAdmin && option.showForNonAdmin) {
          label = isAdmin ? option.showForAdmin : option.showForNonAdmin;
        }
        if (option.descriptionForAdmin && option.descriptionForNonAdmin) {
          desc = isAdmin ? option.descriptionForAdmin : option.descriptionForNonAdmin;
        }
        return {
          value: option.value,
          label,
          description: desc,
          icon: option.icon,
          badgeText: option.badgeText,
        };
      });
  };
  const filteredOptions = getFilteredOptions();
  return (
    <div className="w-full" data-id={dataId}>
      <div className="flex flex-wrap items-center justify-start gap-4">
        <div className="flex items-stretch gap-2 min-w-0">
          <div className="flex-shrink-0 flex items-center">
            <BarChartOutlined style={{ fontSize: "32px" }} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-gray-900 mb-0.5 leading-tight">{resolvedTitle}</h3>
            <p className="text-xs text-gray-600 leading-tight">{resolvedDescription}</p>
          </div>
        </div>
        <div className="flex-shrink-0">
          <Select
            value={value}
            onChange={onChange}
            className="w-54 sm:w-64 md:w-72"
            size="large"
            options={filteredOptions.map((opt) => ({
              value: opt.value,
              label: opt.label,
            }))}
            optionRender={(option) => {
              const opt = filteredOptions.find((o) => o.value === option.value);
              if (!opt) return option.label;
              return (
                <div className="flex items-center gap-2 py-1">
                  <div className="flex-shrink-0 mt-0.5">{opt.icon}</div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-900">{opt.label}</div>
                    <div className="text-xs text-gray-600 mt-0.5">{opt.description}</div>
                  </div>
                  {opt.badgeText && (
                    <div className="items-center">
                      <Badge color="blue" count={opt.badgeText} />
                    </div>
                  )}
                </div>
              );
            }}
            labelRender={(props) => {
              const opt = filteredOptions.find((o) => o.value === props.value);
              if (!opt) return props.label;
              return (
                <div className="flex items-center gap-2">
                  <div>{opt.icon}</div>
                  <span className="text-sm">{opt.label}</span>
                </div>
              );
            }}
          />
        </div>
      </div>
    </div>
  );
};

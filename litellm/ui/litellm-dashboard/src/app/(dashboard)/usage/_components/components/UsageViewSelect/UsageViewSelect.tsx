import { useTerminology } from "@/contexts/TerminologyContext";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { BarChart3, Bot, Building2, Globe, LineChart, ShoppingCart, Tags, User, Users } from "lucide-react";
import React from "react";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import i18n from "@/i18n/i18n";
import { hasCapability, type Capability } from "@/utils/capabilities";
import { all_admin_roles } from "@/utils/roles";
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
  userRole: string | null;
  canViewTagUsage?: boolean;
  isOrgAdmin?: boolean;
  title?: string;
  description?: string;
  "data-id"?: string;
}
interface OptionConfig {
  value: UsageOption;
  label: ParseKeys;
  description: ParseKeys;
  icon: React.ReactNode;
  capability?: Capability;
  adminOnly?: boolean;
  showForAdmin?: ParseKeys;
  showForNonAdmin?: ParseKeys;
  descriptionForAdmin?: ParseKeys;
  descriptionForNonAdmin?: ParseKeys;
  badgeText?: string;
}
const OPTIONS: OptionConfig[] = [
  {
    value: "global",
    label: "usage.viewSelect.options.global.label",
    showForAdmin: "usage.viewSelect.options.global.adminLabel",
    showForNonAdmin: "usage.viewSelect.options.global.nonAdminLabel",
    description: "usage.viewSelect.options.global.description",
    descriptionForAdmin: "usage.viewSelect.options.global.adminDescription",
    descriptionForNonAdmin: "usage.viewSelect.options.global.nonAdminDescription",
    icon: <Globe className="size-4" />,
  },
  {
    value: "my-usage",
    label: "usage.viewSelect.options.myUsage.label",
    description: "usage.viewSelect.options.myUsage.description",
    icon: <User className="size-4" />,
    adminOnly: true,
  },
  {
    value: "organization",
    label: "usage.viewSelect.options.organization.label",
    description: "usage.viewSelect.options.organization.description",
    icon: <Building2 className="size-4" />,
    capability: "viewOrganizationUsage",
  },
  {
    value: "team",
    label: "usage.viewSelect.options.team.label",
    description: "usage.viewSelect.options.team.description",
    icon: <Users className="size-4" />,
  },
  {
    value: "customer",
    label: "usage.viewSelect.options.customer.label",
    description: "usage.viewSelect.options.customer.description",
    icon: <ShoppingCart className="size-4" />,
    adminOnly: true,
  },
  {
    value: "tag",
    label: "usage.viewSelect.options.tag.label",
    description: "usage.viewSelect.options.tag.description",
    icon: <Tags className="size-4" />,
    adminOnly: true,
  },
  {
    value: "agent",
    label: "usage.viewSelect.options.agent.label",
    description: "usage.viewSelect.options.agent.description",
    icon: <Bot className="size-4" />,
    capability: "viewAgentUsage",
  },
  {
    value: "user",
    label: "usage.viewSelect.options.user.label",
    description: "usage.viewSelect.options.user.description",
    icon: <User className="size-4" />,
    adminOnly: true,
  },
  {
    value: "user-agent-activity",
    label: "usage.viewSelect.options.userAgentActivity.label",
    description: "usage.viewSelect.options.userAgentActivity.description",
    icon: <LineChart className="size-4" />,
    adminOnly: true,
  },
];
const visibleOptions = (userRole: string | null, canViewTagUsage: boolean, isOrgAdmin: boolean): OptionConfig[] => {
  const isAdmin = all_admin_roles.includes(userRole ?? "");
  return OPTIONS.filter((option) => {
    if (option.capability) {
      return hasCapability(userRole, option.capability, isOrgAdmin);
    }
    if (option.value === "tag" && canViewTagUsage) {
      return true;
    }
    return !(option.adminOnly && !isAdmin);
  });
};

export const USAGE_OPTIONS: readonly UsageOption[] = OPTIONS.map((option) => option.value);

export const allowedUsageOptions = (
  userRole: string | null,
  canViewTagUsage: boolean,
  isOrgAdmin: boolean,
): readonly UsageOption[] => visibleOptions(userRole, canViewTagUsage, isOrgAdmin).map((option) => option.value);

export const UsageViewSelect: React.FC<UsageViewSelectProps> = ({
  value,
  onChange,
  userRole,
  canViewTagUsage = false,
  isOrgAdmin = false,
  title = i18n.t("usage.viewSelect.title"),
  description = i18n.t("usage.viewSelect.description"),
  "data-id": dataId,
}) => {
  const { t } = useTranslation();
  const term = useTerminology();
  const isAdmin = all_admin_roles.includes(userRole ?? "");
  const getFilteredOptions = () => {
    return visibleOptions(userRole, canViewTagUsage, isOrgAdmin).map((option) => {
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
        label: t(label),
        description: term(t(desc)),
        icon: option.icon,
        badgeText: option.badgeText,
      };
    });
  };
  const filteredOptions = getFilteredOptions();
  const selectedOption = filteredOptions.find((option) => option.value === value);
  return (
    <div className="flex items-center gap-3" data-id={dataId}>
      <h1 className="flex items-center gap-1.5 text-sm font-semibold text-foreground">
        <BarChart3 aria-hidden="true" className="size-4" strokeWidth={2} />
        {title}
      </h1>
      <Select
        value={value}
        onValueChange={(next: UsageOption | null) => {
          if (next) onChange(next);
        }}
      >
        <SelectTrigger aria-label={description} className="h-8 w-52">
          <SelectValue>
            {selectedOption && (
              <span className="flex items-center gap-2">
                {selectedOption.icon}
                <span className="text-sm">{selectedOption.label}</span>
              </span>
            )}
          </SelectValue>
        </SelectTrigger>
        {/* Sized to its content rather than the narrow trigger, so descriptions never clip. */}
        <SelectContent align="start" className="w-auto min-w-72 p-1">
          {filteredOptions.map((option) => (
            <SelectItem key={option.value} value={option.value} className="rounded-md py-2 pr-8 pl-2">
              <span className="flex items-start gap-2.5">
                <span className="mt-0.5 shrink-0 text-muted-foreground">{option.icon}</span>
                <span className="min-w-0">
                  <span className="block text-sm leading-5 text-foreground">{option.label}</span>
                  <span className="block text-xs leading-4 text-muted-foreground">{option.description}</span>
                </span>
                {option.badgeText && <Badge>{option.badgeText}</Badge>}
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

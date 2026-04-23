import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart3,
  Bot,
  Building2,
  Globe,
  LineChart,
  ShoppingCart,
  Tags,
  User,
  Users,
} from "lucide-react";
import React from "react";

export type UsageOption =
  | "global"
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

const ICON_CLASS = "h-4 w-4";

const OPTIONS: OptionConfig[] = [
  {
    value: "global",
    label: "Global Usage",
    showForAdmin: "Global Usage",
    showForNonAdmin: "Your Usage",
    description: "View usage across all resources",
    descriptionForAdmin: "View usage across all resources",
    descriptionForNonAdmin: "View your usage",
    icon: <Globe className={ICON_CLASS} />,
  },
  {
    value: "organization",
    label: "Organization Usage",
    showForAdmin: "Organization Usage",
    showForNonAdmin: "Your Organization Usage",
    description: "View organization-level usage",
    descriptionForAdmin: "View usage across all organizations",
    descriptionForNonAdmin: "View your organization's usage",
    icon: <Building2 className={ICON_CLASS} />,
  },
  {
    value: "team",
    label: "Team Usage",
    description: "View usage by team",
    icon: <Users className={ICON_CLASS} />,
  },
  {
    value: "customer",
    label: "Customer Usage",
    description: "View usage by customer accounts",
    icon: <ShoppingCart className={ICON_CLASS} />,
    adminOnly: true,
  },
  {
    value: "tag",
    label: "Tag Usage",
    description: "View usage grouped by tags",
    icon: <Tags className={ICON_CLASS} />,
    adminOnly: true,
  },
  {
    value: "agent",
    label: "Agent Usage (A2A)",
    description: "View usage by AI agents",
    icon: <Bot className={ICON_CLASS} />,
    adminOnly: true,
  },
  {
    value: "user",
    label: "User Usage",
    description: "View usage by individual users",
    icon: <User className={ICON_CLASS} />,
    adminOnly: true,
  },
  {
    value: "user-agent-activity",
    label: "User Agent Activity",
    description: "View detailed user agent activity logs",
    icon: <LineChart className={ICON_CLASS} />,
    adminOnly: true,
  },
];

export const UsageViewSelect: React.FC<UsageViewSelectProps> = ({
  value,
  onChange,
  isAdmin,
  title = "Usage View",
  description = "Select the usage data you want to view",
  "data-id": dataId,
}) => {
  const filteredOptions = OPTIONS.filter((option) => {
    if (option.adminOnly && !isAdmin) return false;
    return true;
  }).map((option) => {
    let label = option.label;
    let desc = option.description;
    if (option.showForAdmin && option.showForNonAdmin) {
      label = isAdmin ? option.showForAdmin : option.showForNonAdmin;
    }
    if (option.descriptionForAdmin && option.descriptionForNonAdmin) {
      desc = isAdmin
        ? option.descriptionForAdmin
        : option.descriptionForNonAdmin;
    }
    return {
      value: option.value,
      label,
      description: desc,
      icon: option.icon,
    };
  });

  return (
    <div className="w-full" data-id={dataId}>
      <div className="flex flex-wrap items-center justify-start gap-4">
        <div className="flex items-stretch gap-2 min-w-0">
          <div className="flex-shrink-0 flex items-center">
            <BarChart3 className="h-8 w-8" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-foreground mb-0.5 leading-tight">
              {title}
            </h3>
            <p className="text-xs text-muted-foreground leading-tight">
              {description}
            </p>
          </div>
        </div>
        <div className="flex-shrink-0">
          <Select
            value={value}
            onValueChange={(v) => onChange(v as UsageOption)}
          >
            <SelectTrigger className="w-[216px] sm:w-64 md:w-72 h-10">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {filteredOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  <div className="flex items-center gap-2 py-1">
                    <div className="flex-shrink-0 mt-0.5">{opt.icon}</div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-foreground">
                        {opt.label}
                      </div>
                      <div className="text-xs text-muted-foreground mt-0.5">
                        {opt.description}
                      </div>
                    </div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
};

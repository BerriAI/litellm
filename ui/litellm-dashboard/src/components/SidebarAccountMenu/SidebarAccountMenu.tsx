import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableBouncingIcon } from "@/app/(dashboard)/hooks/useDisableBouncingIcon";
import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { useDisableUsageIndicator } from "@/app/(dashboard)/hooks/useDisableUsageIndicator";
import { emitLocalStorageChange, removeLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";
import { navAccountDisplayName } from "@/components/Navbar/navDisplayName";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/cva.config";
import { Check, ChevronsUpDown, Copy, Crown, IdCard, LogOut, Mail, ShieldCheck } from "lucide-react";
import React, { useEffect, useState } from "react";

const RELEASE_NOTES_URL = "https://docs.litellm.ai/release_notes";

function hueFromString(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i += 1) {
    h = seed.charCodeAt(i) + ((h << 5) - h);
  }
  return Math.abs(h) % 360;
}

function initialsFromIdentity(email: string | null, userId: string | null): string {
  const local = email?.split("@")[0]?.trim();
  if (local) {
    const parts = local
      .replace(/[^a-zA-Z0-9]+/g, " ")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0]!.charAt(0)}${parts[1]!.charAt(0)}`.toUpperCase();
    }
    if (parts.length === 1) {
      const p = parts[0]!;
      return p.length >= 2 ? p.slice(0, 2).toUpperCase() : `${p.charAt(0)}`.toUpperCase();
    }
  }
  if (userId && userId.length >= 2) {
    return userId.slice(0, 2).toUpperCase();
  }
  if (userId && userId.length === 1) {
    return `${userId.toUpperCase()}•`;
  }
  return "?";
}

const CopyButton: React.FC<{ value: string | null; label: string }> = ({ value, label }) => {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1200);
    return () => clearTimeout(timer);
  }, [copied]);

  if (!value) return null;

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-xs"
      onClick={() => {
        try {
          navigator.clipboard?.writeText(value);
          setCopied(true);
        } catch {
          setCopied(false);
        }
      }}
      aria-label={label}
      title={label}
      className="text-muted-foreground hover:text-primary"
    >
      {copied ? <Check className="size-[15px]" /> : <Copy className="size-[15px]" />}
    </Button>
  );
};

const InfoRow: React.FC<{ icon: React.ReactNode; label: string; children: React.ReactNode }> = ({
  icon,
  label,
  children,
}) => (
  <div className="flex min-h-[34px] items-center justify-between gap-3">
    <span className="flex items-center gap-2 text-[13px] text-muted-foreground">
      {icon}
      {label}
    </span>
    {children}
  </div>
);

const MonoValue: React.FC<{ value: string | null; copyLabel: string }> = ({ value, copyLabel }) => (
  <span className="flex min-w-0 items-center gap-1">
    <span className="max-w-[150px] truncate font-mono text-[13px] font-medium text-foreground" title={value || "-"}>
      {value || "-"}
    </span>
    <CopyButton value={value} label={copyLabel} />
  </span>
);

interface SidebarAccountMenuProps {
  onLogout: () => void;
  collapsed?: boolean;
}

const SidebarAccountMenu: React.FC<SidebarAccountMenuProps> = ({ onLogout, collapsed = false }) => {
  const { userId, userEmail, userRole, premiumUser, accessToken } = useAuthorized();
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const version = healthData?.litellm_version;
  const disableShowPrompts = useDisableShowPrompts();
  const disableUsageIndicator = useDisableUsageIndicator();
  const disableBlogPosts = useDisableBlogPosts();
  const disableBouncingIcon = useDisableBouncingIcon();
  const disableShowNewBadge = useDisableShowNewBadge();

  const setFlag = (key: string, checked: boolean) => {
    if (checked) {
      setLocalStorageItem(key, "true");
    } else {
      removeLocalStorageItem(key);
    }
    emitLocalStorageChange(key);
  };

  const toggles = [
    {
      key: "disableShowNewBadge",
      label: "Hide New Feature Indicators",
      ariaLabel: "Toggle hide new feature indicators",
      checked: disableShowNewBadge,
      onCheckedChange: (checked: boolean) => setFlag("disableShowNewBadge", checked),
    },
    {
      key: "disableShowPrompts",
      label: "Hide All Prompts",
      ariaLabel: "Toggle hide all prompts",
      checked: disableShowPrompts,
      onCheckedChange: (checked: boolean) => setFlag("disableShowPrompts", checked),
    },
    {
      key: "disableUsageIndicator",
      label: "Hide Usage Indicator",
      ariaLabel: "Toggle hide usage indicator",
      checked: disableUsageIndicator,
      onCheckedChange: (checked: boolean) => setFlag("disableUsageIndicator", checked),
    },
    {
      key: "disableBlogPosts",
      label: "Hide Blog Posts",
      ariaLabel: "Toggle hide blog posts",
      checked: disableBlogPosts,
      onCheckedChange: (checked: boolean) => setFlag("disableBlogPosts", checked),
    },
    {
      key: "disableBouncingIcon",
      label: "Hide Bouncing Icon",
      ariaLabel: "Toggle hide bouncing icon",
      checked: disableBouncingIcon,
      onCheckedChange: (checked: boolean) => setFlag("disableBouncingIcon", checked),
    },
  ];

  const seed = userEmail || userId || "user";
  const initials = initialsFromIdentity(userEmail, userId);
  const hue = hueFromString(seed);
  const displayName = navAccountDisplayName(userEmail, userId);
  const triggerLabel = `Account menu — ${userRole ?? "Unknown role"} — signed in as ${userEmail || userId || "unknown"}`;

  return (
    <Popover>
      <PopoverTrigger
        className={cn(
          "flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",
          collapsed ? "justify-center px-0 py-1" : "gap-2.5 px-2 py-1.5 text-left",
        )}
        aria-label={triggerLabel}
        title={collapsed ? displayName : undefined}
      >
        <Avatar className="size-[30px] shadow-inner ring-1 ring-black/5" aria-hidden>
          <AvatarFallback className="font-semibold text-white" style={{ backgroundColor: `hsl(${hue} 46% 38%)` }}>
            {initials}
          </AvatarFallback>
        </Avatar>
        {!collapsed && (
          <>
            <span className="min-w-0 flex-1 leading-tight">
              <span className="block truncate text-[13px] font-medium text-sidebar-foreground">{displayName}</span>
              {userRole && <span className="block truncate text-[11px] text-muted-foreground">{userRole}</span>}
            </span>
            <ChevronsUpDown size={16} strokeWidth={1.75} className="shrink-0 text-muted-foreground" aria-hidden />
          </>
        )}
      </PopoverTrigger>

      <PopoverContent
        side="top"
        align="start"
        sideOffset={8}
        className="w-[268px] gap-0 overflow-hidden p-0"
        data-testid="sidebar-account-menu-panel"
      >
        <div className="flex items-center gap-2 border-b border-border px-3 py-3">
          <span className="text-[15px] font-bold tracking-tight text-foreground">LiteLLM</span>
          {!disableBouncingIcon && (
            <span
              className="animate-bounce text-lg leading-none"
              style={{ animationDuration: "2s" }}
              title="Thanks for using LiteLLM!"
              aria-hidden
            >
              🌑
            </span>
          )}
          <span className="flex-1" />
          {version && (
            <Badge
              variant="outline"
              render={<a href={RELEASE_NOTES_URL} target="_blank" rel="noopener noreferrer" />}
              className="px-1.5 py-0 font-mono text-[10px] font-medium text-muted-foreground"
            >
              v{version}
            </Badge>
          )}
        </div>

        <div className="flex flex-col px-3 py-2">
          <InfoRow icon={<Crown className="size-[17px]" />} label="Tier">
            {premiumUser ? (
              <Badge
                variant="outline"
                className="gap-1 border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-500"
              >
                <Crown />
                Premium
              </Badge>
            ) : (
              <Badge variant="secondary" className="gap-1" title="Upgrade to Premium for advanced features">
                <Crown />
                Standard
              </Badge>
            )}
          </InfoRow>
          <InfoRow icon={<ShieldCheck className="size-[17px]" />} label="Role">
            <Badge variant="secondary">{userRole}</Badge>
          </InfoRow>
          <InfoRow icon={<Mail className="size-[17px]" />} label="Email">
            <MonoValue value={userEmail} copyLabel="Copy email" />
          </InfoRow>
          <InfoRow icon={<IdCard className="size-[17px]" />} label="User ID">
            <MonoValue value={userId} copyLabel="Copy user ID" />
          </InfoRow>
        </div>

        <Separator />

        <div className="py-1">
          {toggles.map((toggle) => (
            <div key={toggle.key} className="flex h-[38px] items-center justify-between gap-3 px-3">
              <span className="text-[13px] text-foreground">{toggle.label}</span>
              <Switch
                size="sm"
                checked={toggle.checked}
                onCheckedChange={toggle.onCheckedChange}
                aria-label={toggle.ariaLabel}
              />
            </div>
          ))}
        </div>

        <Separator />

        <Button
          variant="ghost"
          onClick={onLogout}
          className="h-[42px] w-full justify-start gap-2.5 rounded-none px-3 text-sm font-medium text-foreground"
        >
          <LogOut className="size-[19px] text-muted-foreground" />
          Logout
        </Button>
      </PopoverContent>
    </Popover>
  );
};

export default SidebarAccountMenu;

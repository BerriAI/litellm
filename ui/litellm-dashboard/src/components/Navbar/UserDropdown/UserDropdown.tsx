import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableBouncingIcon } from "@/app/(dashboard)/hooks/useDisableBouncingIcon";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import {
  emitLocalStorageChange,
  getLocalStorageItem,
  removeLocalStorageItem,
  setLocalStorageItem,
} from "@/utils/localStorageUtils";
import { navAccountDisplayName } from "@/components/Navbar/navDisplayName";
import { ChevronDown, ChevronsUpDown, Crown, LogOut, Mail, ShieldCheck, User } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import CopyButton from "@/components/shared/CopyButton";
import { cn } from "@/lib/cva.config";
import React, { useEffect, useState } from "react";

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

interface UserDropdownProps {
  onLogout: () => void;
  // "navbar" (default): compact top-right trigger. "sidebar": full-width footer
  // trigger whose menu opens upward, for the redesigned sidebar dock.
  variant?: "navbar" | "sidebar";
  // Sidebar rail mode: render the avatar only (no name/role).
  collapsed?: boolean;
}

const UserDropdown: React.FC<UserDropdownProps> = ({ onLogout, variant = "navbar", collapsed = false }) => {
  const { userId, userEmail, userRoleLabel: userRole, premiumUser } = useAuthorized();
  const disableShowPrompts = useDisableShowPrompts();
  const disableBlogPosts = useDisableBlogPosts();
  const disableBouncingIcon = useDisableBouncingIcon();
  const [disableShowNewBadge, setDisableShowNewBadge] = useState(false);

  useEffect(() => {
    const storedValue = getLocalStorageItem("disableShowNewBadge");
    setDisableShowNewBadge(storedValue === "true");
  }, []);

  const renderUserInfoSection = () => (
    <div className="flex w-full flex-col gap-2 p-3 text-sm">
      <div className="flex w-full items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Mail className="size-4" />
          <span className="text-muted-foreground">{userEmail || "-"}</span>
        </div>
        {premiumUser ? (
          <Badge>
            <Crown className="size-3" />
            Premium
          </Badge>
        ) : (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger render={<Badge variant="outline" />}>
                <Crown className="size-3" />
                Standard
              </TooltipTrigger>
              <TooltipContent side="left">Upgrade to Premium for advanced features</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      <Separator className="my-2" />
      <div className="flex w-full items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <User className="size-4" />
          <span className="text-muted-foreground">User ID</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="max-w-[150px] truncate" title={userId || "-"}>
            {userId || "-"}
          </span>
          <CopyButton value={userId} label="Copy User ID" />
        </div>
      </div>
      <div className="flex w-full items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-4" />
          <span className="text-muted-foreground">Role</span>
        </div>
        <span>{userRole}</span>
      </div>
      <Separator className="my-2" />
      <div className="flex w-full items-center justify-between gap-2">
        <span className="text-muted-foreground">Hide New Feature Indicators</span>
        <Switch
          size="sm"
          checked={disableShowNewBadge}
          onCheckedChange={(checked) => {
            setDisableShowNewBadge(checked);
            if (checked) {
              setLocalStorageItem("disableShowNewBadge", "true");
              emitLocalStorageChange("disableShowNewBadge");
            } else {
              removeLocalStorageItem("disableShowNewBadge");
              emitLocalStorageChange("disableShowNewBadge");
            }
          }}
          aria-label="Toggle hide new feature indicators"
        />
      </div>
      <div className="flex w-full items-center justify-between gap-2">
        <span className="text-muted-foreground">Hide All Prompts</span>
        <Switch
          size="sm"
          checked={disableShowPrompts}
          onCheckedChange={(checked) => {
            if (checked) {
              setLocalStorageItem("disableShowPrompts", "true");
              emitLocalStorageChange("disableShowPrompts");
            } else {
              removeLocalStorageItem("disableShowPrompts");
              emitLocalStorageChange("disableShowPrompts");
            }
          }}
          aria-label="Toggle hide all prompts"
        />
      </div>
      <div className="flex w-full items-center justify-between gap-2">
        <span className="text-muted-foreground">Hide Blog Posts</span>
        <Switch
          size="sm"
          checked={disableBlogPosts}
          onCheckedChange={(checked) => {
            if (checked) {
              setLocalStorageItem("disableBlogPosts", "true");
              emitLocalStorageChange("disableBlogPosts");
            } else {
              removeLocalStorageItem("disableBlogPosts");
              emitLocalStorageChange("disableBlogPosts");
            }
          }}
          aria-label="Toggle hide blog posts"
        />
      </div>
      <div className="flex w-full items-center justify-between gap-2">
        <span className="text-muted-foreground">Hide Bouncing Icon</span>
        <Switch
          size="sm"
          checked={disableBouncingIcon}
          onCheckedChange={(checked) => {
            if (checked) {
              setLocalStorageItem("disableBouncingIcon", "true");
              emitLocalStorageChange("disableBouncingIcon");
            } else {
              removeLocalStorageItem("disableBouncingIcon");
              emitLocalStorageChange("disableBouncingIcon");
            }
          }}
          aria-label="Toggle hide bouncing icon"
        />
      </div>
    </div>
  );

  const seed = userEmail || userId || "user";
  const initials = initialsFromIdentity(userEmail, userId);
  const hue = hueFromString(seed);
  const displayName = navAccountDisplayName(userEmail, userId);

  return (
    <Popover>
      {variant === "sidebar" ? (
        <PopoverTrigger
          render={
            <button
              type="button"
              className={cn(
                "flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",
                collapsed ? "justify-center px-0 py-1" : "gap-2.5 px-2 py-1.5 text-left",
              )}
              aria-label={`Account menu — ${userRole ?? "Unknown role"} — signed in as ${userEmail || userId || "unknown"}`}
              aria-haspopup="dialog"
              title={collapsed ? displayName : undefined}
            />
          }
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
      ) : (
        <PopoverTrigger
          render={
            <button
              type="button"
              className="flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-accent!"
              aria-label={`Account menu — ${userRole ?? "Unknown role"} — signed in as ${userEmail || userId || "unknown"}`}
              aria-haspopup="dialog"
            />
          }
        >
          <Avatar className="shadow-inner ring-1 ring-black/5" aria-hidden>
            <AvatarFallback className="font-semibold text-white" style={{ backgroundColor: `hsl(${hue} 46% 38%)` }}>
              {initials}
            </AvatarFallback>
          </Avatar>
          <span className="hidden min-w-0 truncate text-left text-sm font-medium leading-none text-foreground md:inline">
            {displayName}
          </span>
          <ChevronDown className="hidden size-2.5 shrink-0 text-muted-foreground md:inline" aria-hidden />
        </PopoverTrigger>
      )}
      <PopoverContent
        align={variant === "sidebar" ? "start" : "end"}
        side={variant === "sidebar" ? "top" : "bottom"}
        className="w-auto gap-0 rounded-lg bg-card p-1 shadow-lg"
        data-testid="user-dropdown-panel"
      >
        {renderUserInfoSection()}
        <Separator />
        <button
          type="button"
          onClick={onLogout}
          className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent"
        >
          <LogOut className="size-4" />
          Logout
        </button>
      </PopoverContent>
    </Popover>
  );
};

export default UserDropdown;

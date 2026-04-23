import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableBouncingIcon } from "@/app/(dashboard)/hooks/useDisableBouncingIcon";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { useDisableUsageIndicator } from "@/app/(dashboard)/hooks/useDisableUsageIndicator";
import {
  emitLocalStorageChange,
  getLocalStorageItem,
  removeLocalStorageItem,
  setLocalStorageItem,
} from "@/utils/localStorageUtils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChevronDown,
  Crown,
  LogOut,
  Mail,
  Shield,
  User,
} from "lucide-react";
import React, { useEffect, useState } from "react";

interface UserDropdownProps {
  onLogout: () => void;
}

const UserDropdown: React.FC<UserDropdownProps> = ({ onLogout }) => {
  const { userId, userEmail, userRole, premiumUser } = useAuthorized();
  const disableShowPrompts = useDisableShowPrompts();
  const disableUsageIndicator = useDisableUsageIndicator();
  const disableBlogPosts = useDisableBlogPosts();
  const disableBouncingIcon = useDisableBouncingIcon();
  const [disableShowNewBadge, setDisableShowNewBadge] = useState(false);

  useEffect(() => {
    const storedValue = getLocalStorageItem("disableShowNewBadge");
    setDisableShowNewBadge(storedValue === "true");
  }, []);

  const toggleLocalStorage = (key: string, checked: boolean) => {
    if (checked) {
      setLocalStorageItem(key, "true");
    } else {
      removeLocalStorageItem(key);
    }
    emitLocalStorageChange(key);
  };

  const ToggleRow: React.FC<{
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    ariaLabel?: string;
  }> = ({ label, checked, onChange, ariaLabel }) => (
    <div className="flex items-center justify-between w-full">
      <span className="text-muted-foreground text-sm">{label}</span>
      <Switch
        checked={checked}
        onCheckedChange={onChange}
        aria-label={ariaLabel ?? label}
      />
    </div>
  );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="gap-2">
          <User className="h-4 w-4" />
          <span className="text-sm">User</span>
          <ChevronDown className="h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72 p-3">
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm">
              <Mail className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground truncate max-w-[160px]">
                {userEmail || "-"}
              </span>
            </div>
            {premiumUser ? (
              <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300 gap-1">
                <Crown className="h-3 w-3" />
                Premium
              </Badge>
            ) : (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="gap-1">
                      <Crown className="h-3 w-3" />
                      Standard
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    Upgrade to Premium for advanced features
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
          <DropdownMenuSeparator />
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm">
              <User className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">User ID</span>
            </div>
            <span
              className="text-sm truncate max-w-[150px]"
              title={userId || "-"}
            >
              {userId || "-"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm">
              <Shield className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Role</span>
            </div>
            <span className="text-sm">{userRole}</span>
          </div>
          <DropdownMenuSeparator />
          <ToggleRow
            label="Hide New Feature Indicators"
            checked={disableShowNewBadge}
            onChange={(c) => {
              setDisableShowNewBadge(c);
              toggleLocalStorage("disableShowNewBadge", c);
            }}
            ariaLabel="Toggle hide new feature indicators"
          />
          <ToggleRow
            label="Hide All Prompts"
            checked={disableShowPrompts}
            onChange={(c) => toggleLocalStorage("disableShowPrompts", c)}
            ariaLabel="Toggle hide all prompts"
          />
          <ToggleRow
            label="Hide Usage Indicator"
            checked={disableUsageIndicator}
            onChange={(c) => toggleLocalStorage("disableUsageIndicator", c)}
            ariaLabel="Toggle hide usage indicator"
          />
          <ToggleRow
            label="Hide Blog Posts"
            checked={disableBlogPosts}
            onChange={(c) => toggleLocalStorage("disableBlogPosts", c)}
            ariaLabel="Toggle hide blog posts"
          />
          <ToggleRow
            label="Hide Bouncing Icon"
            checked={disableBouncingIcon}
            onChange={(c) => toggleLocalStorage("disableBouncingIcon", c)}
            ariaLabel="Toggle hide bouncing icon"
          />
        </div>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onLogout}>
          <LogOut className="h-4 w-4" />
          Logout
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default UserDropdown;

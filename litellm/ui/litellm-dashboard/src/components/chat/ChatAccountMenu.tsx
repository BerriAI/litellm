"use client";

import { useTranslation } from "react-i18next";
import { ChevronsUpDown, LogOut, Settings } from "lucide-react";
import { useChatShell } from "@/contexts/ChatShellContext";
import { navAccountDisplayName } from "@/components/Navbar/navDisplayName";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

export default function ChatAccountMenu({
  collapsed,
  onSettings,
  onLogout,
}: {
  collapsed: boolean;
  onSettings: () => void;
  onLogout: () => void;
}) {
  const { t } = useTranslation();
  const { userEmail, userId } = useChatShell();
  const identity = userEmail || userId;
  const name = navAccountDisplayName(userEmail, userId);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button
            variant="ghost"
            aria-label={t("chat.shell.accountMenu", { identity })}
            className={cn(
              "h-auto w-full gap-2.5 rounded-lg hover:bg-sidebar-accent",
              collapsed ? "justify-center px-0 py-1" : "justify-start px-2 py-1.5",
            )}
          />
        }
      >
        <Avatar className="size-[30px]">
          <AvatarFallback className="bg-sidebar-accent text-sidebar-accent-foreground text-xs font-medium">
            {name.slice(0, 2).toUpperCase()}
          </AvatarFallback>
        </Avatar>
        {!collapsed && (
          <>
            <span className="min-w-0 flex-1 truncate text-left text-[13px] font-medium">{name}</span>
            <ChevronsUpDown className="size-4 text-muted-foreground" />
          </>
        )}
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="start" className="w-60">
        <div className="truncate px-2 py-2 text-xs text-muted-foreground" title={identity}>
          {identity}
        </div>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onSettings}>
          <Settings />
          {t("chat.shell.settings")}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={onLogout}>
          <LogOut />
          {t("chat.shell.signOut")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

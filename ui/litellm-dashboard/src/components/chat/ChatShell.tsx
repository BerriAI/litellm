"use client";

import React, { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  Pencil,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  MessageSquare,
  LayoutGrid,
  KeyRound,
  Lock,
  BarChart3,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { migratedHref } from "@/utils/migratedPages";
import { useChatShell } from "@/contexts/ChatShellContext";
import ConversationList from "./ConversationList";

const CHAT_BASE = migratedHref("chat");
export const CHAT_ROUTES = {
  chats: CHAT_BASE,
  integrations: `${CHAT_BASE}/integrations`,
  credentials: `${CHAT_BASE}/credentials`,
  apiKeys: `${CHAT_BASE}/api-keys`,
  usage: `${CHAT_BASE}/usage`,
};

function stripTrailingSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
  collapsed: boolean;
}

function NavItem({ icon, label, onClick, active = false, collapsed }: NavItemProps) {
  const btn = (
    <button
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={`flex items-center gap-2.5 px-2.5 py-2 w-full rounded-md border-none text-sm text-left transition-colors ${
        collapsed ? "justify-center" : "justify-start"
      } ${active ? "bg-accent text-accent-foreground" : "text-foreground/70 hover:bg-accent/50"}`}
      style={{ cursor: "pointer" }}
    >
      <span className="shrink-0">{icon}</span>
      {!collapsed && <span className="flex-1 text-left font-medium">{label}</span>}
    </button>
  );
  if (!collapsed) return btn;
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>{btn}</TooltipTrigger>
        <TooltipContent side="right">
          <p>{label}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface ChatShellProps {
  children: React.ReactNode;
}

const ChatShell: React.FC<ChatShellProps> = ({ children }) => {
  const router = useRouter();
  const pathname = stripTrailingSlash(usePathname() ?? "");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { conversations, activeConversationId, deleteConversation, renameConversation } = useChatShell();

  const isChatsRoute = pathname === CHAT_ROUTES.chats;

  return (
    <div className="flex h-full w-full bg-background overflow-hidden">
      <div
        className="shrink-0 bg-secondary border-r flex flex-col overflow-hidden"
        style={{ width: sidebarCollapsed ? 56 : 260, transition: "width 0.2s cubic-bezier(0.4, 0, 0.2, 1)" }}
      >
        <div className="flex items-center justify-start px-2.5 py-3 shrink-0">
          <TooltipProvider delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => setSidebarCollapsed((v) => !v)}
                  className="p-1.5 rounded-md text-muted-foreground hover:text-foreground flex items-center cursor-pointer transition-colors"
                  style={{ background: "none", border: "none" }}
                >
                  {sidebarCollapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
                </button>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>{sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        <div className="px-2 pb-1 shrink-0">
          <NavItem
            icon={<Pencil className="h-4 w-4" />}
            label="New chat"
            onClick={() => router.push(CHAT_ROUTES.chats)}
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<Search className="h-4 w-4" />}
            label="Search chats"
            onClick={() => router.push(CHAT_ROUTES.chats)}
            collapsed={sidebarCollapsed}
          />
        </div>

        <Separator className="mx-2 shrink-0" />

        <div className="px-2 py-1 shrink-0">
          <NavItem
            icon={<MessageSquare className="h-4 w-4" />}
            label="Chats"
            onClick={() => router.push(CHAT_ROUTES.chats)}
            active={isChatsRoute}
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<LayoutGrid className="h-4 w-4" />}
            label="Integrations"
            onClick={() => router.push(CHAT_ROUTES.integrations)}
            active={pathname === CHAT_ROUTES.integrations}
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<KeyRound className="h-4 w-4" />}
            label="Credentials"
            onClick={() => router.push(CHAT_ROUTES.credentials)}
            active={pathname === CHAT_ROUTES.credentials}
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<Lock className="h-4 w-4" />}
            label="API Keys"
            onClick={() => router.push(CHAT_ROUTES.apiKeys)}
            active={pathname === CHAT_ROUTES.apiKeys}
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<BarChart3 className="h-4 w-4" />}
            label="Usage"
            onClick={() => router.push(CHAT_ROUTES.usage)}
            active={pathname === CHAT_ROUTES.usage}
            collapsed={sidebarCollapsed}
          />
        </div>

        <Separator className="mx-2 shrink-0" />

        {!sidebarCollapsed && (
          <div className="flex-1 overflow-hidden flex flex-col">
            <ConversationList
              conversations={conversations}
              activeConversationId={activeConversationId}
              onSelect={(id) => router.push(`${CHAT_ROUTES.chats}?id=${id}`)}
              onDelete={(id) => {
                deleteConversation(id);
                if (id === activeConversationId) router.push(CHAT_ROUTES.chats);
              }}
              onNewChat={() => router.push(CHAT_ROUTES.chats)}
              onRename={renameConversation}
            />
          </div>
        )}
      </div>

      <div className="flex-1 flex flex-col overflow-hidden min-w-0">{children}</div>
    </div>
  );
};

export default ChatShell;

"use client";

import React, { useState } from "react";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import {
  Plus,
  MessageSquare,
  Network,
  LayoutGrid,
  KeyRound,
  Lock,
  BarChart3,
  ScrollText,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Settings2,
  ChevronDown,
  Info,
  Settings,
  LogOut,
} from "lucide-react";
import { useMediaQuery } from "usehooks-ts";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import ThemeToggle from "@/components/ThemeToggle/ThemeToggle";
import { useLogout } from "@/app/(dashboard)/hooks/useLogout";
import { uiHref } from "@/utils/uiHref";
import { useChatShell } from "@/contexts/ChatShellContext";
import { useTheme } from "@/contexts/ThemeContext";
import ViewSwitcher from "@/components/Navbar/ViewSwitcher";
import ConversationList from "./ConversationList";

export function getChatRoutes() {
  const base = uiHref("chat");
  return {
    chats: base,
    models: `${base}/models`,
    integrations: `${base}/integrations`,
    credentials: `${base}/credentials`,
    apiKeys: `${base}/api-keys`,
    logs: `${base}/logs`,
    usage: `${base}/usage`,
  };
}

function stripTrailingSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
}

function NavItem({ icon, label, onClick, active = false }: NavItemProps) {
  return (
    <Button
      onClick={onClick}
      variant="ghost"
      aria-current={active ? "page" : undefined}
      className={`chat-nav-item moyai-chat-rail-button w-full justify-start ${active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground"}`}
    >
      {icon}
      <span className="flex-1 text-left">{label}</span>
    </Button>
  );
}

const ChatShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const router = useRouter();
  const pathname = stripTrailingSlash(usePathname() ?? "");
  const {
    accessToken,
    userEmail,
    userId,
    conversations,
    activeConversationId,
    deleteConversation,
    renameConversation,
  } = useChatShell();
  const handleLogout = useLogout(accessToken);
  const isDesktop = useMediaQuery("(min-width: 601px)", { initializeWithValue: false });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  if (isDesktop && sidebarOpen) setSidebarOpen(false);
  const collapsed = isDesktop && sidebarCollapsed;
  const { logoUrl, logoUrlDark } = useTheme();
  const lightLogo = logoUrl || logoUrlDark;
  const darkLogo = logoUrlDark || logoUrl;
  const routes = getChatRoutes();
  const isChatsRoute = pathname === routes.chats;
  const accountName = userEmail?.split("@")[0] || userId || "Account";
  const workspaceItems = [
    { label: "Credentials", route: routes.credentials, Icon: KeyRound },
    { label: "API Keys", route: routes.apiKeys, Icon: Lock },
    { label: "Logs", route: routes.logs, Icon: ScrollText },
    { label: "Usage", route: routes.usage, Icon: BarChart3 },
  ];
  const workspaceActive = workspaceItems.some((item) => item.route === pathname);
  const navigate = (href: string) => {
    setSidebarOpen(false);
    setSearchOpen(false);
    router.push(href);
  };

  const sidebar = (
    <>
      <div className="moyai-chat-brand-row">
        <ViewSwitcher className="moyai-chat-workspace-switcher" contentClassName="chat-theme min-w-60" />
        <Button
          variant="ghost"
          size="icon-sm"
          className="moyai-chat-search"
          aria-label="Search chats"
          title="Search chats (⌘K)"
          onClick={() => setSearchOpen(true)}
        >
          <Search className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={collapsed ? "Open sidebar" : "Close sidebar"}
          aria-expanded={!collapsed}
          onClick={() => (isDesktop ? setSidebarCollapsed(!sidebarCollapsed) : setSidebarOpen(false))}
        >
          {collapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
        </Button>
      </div>
      <Button
        variant="ghost"
        onClick={() => navigate(routes.chats)}
        className="chat-new-conversation moyai-chat-rail-button w-full justify-start"
      >
        <Plus className="size-4" />
        New Chat
      </Button>
      <nav aria-label="Chat workspace" className="moyai-chat-destinations">
        <NavItem
          icon={<MessageSquare className="size-4" />}
          label="Chats"
          onClick={() => navigate(routes.chats)}
          active={isChatsRoute}
        />
        <NavItem
          icon={<Network className="size-4" />}
          label="Models"
          onClick={() => navigate(routes.models)}
          active={pathname === routes.models}
        />
        <NavItem
          icon={<LayoutGrid className="size-4" />}
          label="Integrations"
          onClick={() => navigate(routes.integrations)}
          active={pathname === routes.integrations}
        />
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                variant="ghost"
                className={`chat-nav-item moyai-chat-rail-button w-full justify-start ${workspaceActive ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground"}`}
                aria-current={workspaceActive ? "page" : undefined}
              />
            }
          >
            <Settings2 className="size-4" />
            <span className="flex-1 text-left">Workspace</span>
            <ChevronDown className="size-3.5" />
          </DropdownMenuTrigger>
          <DropdownMenuContent side="right" align="start" className="chat-theme min-w-44">
            {workspaceItems.map(({ label, route, Icon }) => (
              <DropdownMenuItem
                key={route}
                aria-current={pathname === route ? "page" : undefined}
                onClick={() => navigate(route)}
              >
                <Icon className="size-4" />
                {label}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <p className="px-2 py-1 text-xs text-muted-foreground">
              Chat is a pre-v0 preview and may change unexpectedly. Do not use in production.
            </p>
            <DropdownMenuItem
              render={
                <a href="https://github.com/BerriAI/litellm/discussions/32085" target="_blank" rel="noreferrer" />
              }
            >
              <Info className="size-4" />
              Share feedback
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </nav>
      <div className="moyai-chat-history flex min-h-0 flex-1 flex-col overflow-hidden">
        <ConversationList
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelect={(id) => navigate(`${routes.chats}?id=${encodeURIComponent(id)}`)}
          onDelete={(id) => {
            deleteConversation(id);
            if (id === activeConversationId) navigate(routes.chats);
          }}
          onRename={renameConversation}
          searchOpen={searchOpen}
          onSearchOpenChange={setSearchOpen}
          hideSearchButton
        />
      </div>
      {lightLogo && darkLogo && (
        <div className="flex h-12 shrink-0 items-center px-1 pb-2" role="img" aria-label="Organization logo">
          <Image
            src={lightLogo}
            alt=""
            width={160}
            height={32}
            unoptimized
            className="h-8 max-w-full object-contain object-left dark:hidden"
          />
          <Image
            src={darkLogo}
            alt=""
            width={160}
            height={32}
            unoptimized
            className="hidden h-8 max-w-full object-contain object-left dark:block"
          />
        </div>
      )}
      <footer className="moyai-chat-account">
        <span className="min-w-0 flex-1 truncate text-[13px]" title={userEmail || userId}>
          {accountName}
        </span>
        <ThemeToggle />
        <Button variant="ghost" size="icon-sm" aria-label="Settings" onClick={() => navigate(routes.credentials)}>
          <Settings className="size-4" />
        </Button>
        <Button variant="ghost" size="icon-sm" aria-label="Sign out" onClick={handleLogout}>
          <LogOut className="size-4" />
        </Button>
      </footer>
    </>
  );

  return (
    <div
      className={`chat-theme chat-workspace moyai-chat-shell flex h-full w-full overflow-hidden text-foreground ${collapsed ? "moyai-chat-collapsed" : ""}`}
    >
      {isDesktop ? (
        <aside aria-label="Chat navigation" className="chat-glass-sidebar moyai-chat-sidebar">
          {sidebar}
        </aside>
      ) : (
        <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
          <div className="moyai-chat-mobile-header z-chrome flex h-16 shrink-0 items-center gap-2 px-3">
            <SheetTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon-lg"
                  className="moyai-chat-mobile-trigger"
                  aria-label="Open chat navigation"
                />
              }
            >
              <PanelLeftOpen className="size-5" />
            </SheetTrigger>
            <ViewSwitcher className="moyai-chat-workspace-switcher" contentClassName="chat-theme min-w-60" />
          </div>
          <SheetContent
            side="left"
            showCloseButton={false}
            className="chat-theme chat-mobile-sidebar moyai-chat-sidebar"
          >
            <SheetHeader className="sr-only">
              <SheetTitle>LiteLLM Chat</SheetTitle>
              <SheetDescription>Conversations and chat workspace navigation</SheetDescription>
            </SheetHeader>
            {sidebar}
          </SheetContent>
        </Sheet>
      )}
      <main className="chat-glass-pane relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{children}</main>
    </div>
  );
};

export default ChatShell;

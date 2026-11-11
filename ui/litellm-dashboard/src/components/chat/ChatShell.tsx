"use client";

import React, { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  Menu,
  Plus,
  Network,
  LayoutGrid,
  KeyRound,
  Lock,
  BarChart3,
  ScrollText,
  Search,
  Settings2,
  ChevronDown,
  Info,
} from "lucide-react";
import { useMediaQuery } from "usehooks-ts";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sidebar, SidebarContent, SidebarFooter, SidebarMenuButton } from "@/components/shared/Sidebar";
import { SidebarBrand } from "@/components/shared/SidebarBrand";
import ChatAccountMenu from "./ChatAccountMenu";
import { Kbd } from "@/components/ui/kbd";
import { useShortcutLabel } from "@/components/CommandPalette/useShortcutLabel";
import { useHotkeys } from "react-hotkeys-hook";
import { DashboardHeader } from "@/components/DashboardHeader";
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { useLogout } from "@/app/(dashboard)/hooks/useLogout";
import { uiHref } from "@/utils/uiHref";
import { useChatShell } from "@/contexts/ChatShellContext";
import ViewSwitcher from "@/components/Navbar/ViewSwitcher";
import ConversationList from "./ConversationList";

const conversationUrl = (chatsRoute: string, id: string) => `${chatsRoute}?id=${encodeURIComponent(id)}`;

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
  shortcut?: string;
}

function NavItem({ icon, label, onClick, active = false, shortcut }: NavItemProps) {
  return (
    <SidebarMenuButton
      onClick={onClick}
      isActive={active}
      aria-label={label}
      title={label}
      aria-current={active ? "page" : undefined}
    >
      {icon}
      <span className="flex-1 truncate group-data-[collapsed=true]/sidebar:hidden">{label}</span>
      {shortcut && (
        <Kbd className="bg-transparent text-muted-foreground group-data-[collapsed=true]/sidebar:hidden">
          {shortcut}
        </Kbd>
      )}
    </SidebarMenuButton>
  );
}

const ChatShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { t } = useTranslation();
  const router = useRouter();
  const pathname = stripTrailingSlash(usePathname() ?? "");
  const { accessToken, conversations, activeConversationId, deleteConversation, renameConversation } = useChatShell();
  const handleLogout = useLogout(accessToken);
  const isDesktop = useMediaQuery("(min-width: 768px)", { initializeWithValue: false });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  if (isDesktop && sidebarOpen) setSidebarOpen(false);
  const collapsed = isDesktop && sidebarCollapsed;
  const [mobileNavigationHandle] = useState(() => DialogPrimitive.createHandle());
  const shortcut = useShortcutLabel();
  const openSearch = () => {
    setSidebarCollapsed(false);
    if (!isDesktop) setSidebarOpen(true);
    setSearchOpen(true);
  };
  useHotkeys("mod+k", openSearch, { enableOnFormTags: true, preventDefault: true, useKey: true });
  const routes = getChatRoutes();
  const isChatsRoute = pathname === routes.chats;
  const workspaceItems = [
    { label: t("chat.shell.workspaceItems.credentials"), route: routes.credentials, Icon: KeyRound },
    { label: t("chat.shell.workspaceItems.apiKeys"), route: routes.apiKeys, Icon: Lock },
    { label: t("chat.shell.workspaceItems.logs"), route: routes.logs, Icon: ScrollText },
    { label: t("chat.shell.workspaceItems.usage"), route: routes.usage, Icon: BarChart3 },
  ];
  const workspaceActive = workspaceItems.some((item) => item.route === pathname);
  const navigate = (href: string) => {
    setSidebarOpen(false);
    setSearchOpen(false);
    router.push(href);
  };

  const sidebar = (
    <>
      <SidebarBrand
        accessToken={accessToken}
        homeHref={routes.chats}
        homeLabel={t("chat.shell.home")}
        onHomeClick={() => setSidebarOpen(false)}
        collapsed={collapsed}
        onToggleCollapsed={() => (isDesktop ? setSidebarCollapsed(!sidebarCollapsed) : setSidebarOpen(false))}
      />
      <div className="space-y-1 px-3 pt-3 pb-2">
        <ViewSwitcher className="h-9 w-full max-w-none justify-start shadow-none group-data-[collapsed=true]/sidebar:size-9 group-data-[collapsed=true]/sidebar:justify-center group-data-[collapsed=true]/sidebar:p-0 [&_[data-slot=workspace-label]]:flex-1 [&_[data-slot=workspace-label]]:text-left group-data-[collapsed=true]/sidebar:[&_[data-slot=workspace-label]]:hidden" />
        <NavItem icon={<Plus />} label={t("chat.shell.newChat")} onClick={() => navigate(routes.chats)} />
      </div>
      <SidebarContent aria-label={t("chat.shell.workspaceNav")} className="flex-none">
        <NavItem
          icon={<Network className="size-4" />}
          label={t("chat.shell.models")}
          onClick={() => navigate(routes.models)}
          active={pathname === routes.models}
        />
        <NavItem
          icon={<LayoutGrid className="size-4" />}
          label={t("chat.shell.integrations")}
          onClick={() => navigate(routes.integrations)}
          active={pathname === routes.integrations}
        />
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <SidebarMenuButton
                isActive={workspaceActive}
                aria-label={t("chat.shell.workspace")}
                title={t("chat.shell.workspace")}
                aria-current={workspaceActive ? "page" : undefined}
              />
            }
          >
            <Settings2 className="size-4" />
            <span className="flex-1 text-left group-data-[collapsed=true]/sidebar:hidden">
              {t("chat.shell.workspace")}
            </span>
            <ChevronDown className="size-3.5 group-data-[collapsed=true]/sidebar:hidden" />
          </DropdownMenuTrigger>
          <DropdownMenuContent side="right" align="start" className="min-w-44">
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
            <DropdownMenuItem
              render={
                <a href="https://github.com/BerriAI/litellm/discussions/32085" target="_blank" rel="noreferrer" />
              }
            >
              <Info className="size-4" />
              {t("chat.shell.shareFeedback")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <NavItem icon={<Search />} label={t("chat.shell.searchChats")} shortcut={shortcut} onClick={openSearch} />
      </SidebarContent>
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden px-3 pb-3 group-data-[collapsed=true]/sidebar:hidden">
        <ConversationList
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelect={(id) => navigate(conversationUrl(routes.chats, id))}
          onDelete={(id) => {
            deleteConversation(id);
            if (id === activeConversationId) navigate(routes.chats);
          }}
          onRename={renameConversation}
          searchOpen={searchOpen}
          onSearchOpenChange={setSearchOpen}
          hideSearchButton
          enableSearchShortcut={false}
        />
      </div>
      <SidebarFooter className="mt-auto">
        <ChatAccountMenu
          onLogout={handleLogout}
          collapsed={collapsed}
          onSettings={() => navigate(routes.credentials)}
        />
      </SidebarFooter>
    </>
  );

  const titles = [
    { route: routes.chats, label: t("shell.viewSwitcher.chat") },
    { route: routes.models, label: t("chat.shell.models") },
    { route: routes.integrations, label: t("chat.shell.integrations") },
    ...workspaceItems,
  ];
  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId);
  const title =
    isChatsRoute && activeConversation?.messages.length
      ? activeConversation.title
      : titles.find((item) => item.route === pathname)?.label ?? t("shell.viewSwitcher.chat");

  return (
    <div className="chat-theme flex h-full w-full overflow-hidden bg-background text-foreground">
      {isDesktop && (
        <Sidebar aria-label={t("chat.shell.chatNavigation")} collapsed={collapsed}>
          {sidebar}
        </Sidebar>
      )}
      <Sheet handle={mobileNavigationHandle} open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <SheetContent
          side="left"
          initialFocus={searchOpen ? false : undefined}
          showCloseButton={false}
          className="gap-0 p-0 data-[side=left]:w-[min(280px,calc(100vw-3rem))] [&_[data-slot=sidebar]]:w-full"
        >
          <SheetHeader className="sr-only">
            <SheetTitle>{t("chat.shell.sheetTitle")}</SheetTitle>
            <SheetDescription>{t("chat.shell.sheetDescription")}</SheetDescription>
          </SheetHeader>
          <Sidebar>{sidebar}</Sidebar>
        </SheetContent>
      </Sheet>
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <DashboardHeader
          accessToken={accessToken}
          title={title}
          workspace="chat"
          navigationTrigger={
            <SheetTrigger
              handle={mobileNavigationHandle}
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-11 md:hidden"
                  aria-label={t("chat.shell.openChatNavigation")}
                />
              }
            >
              <Menu />
            </SheetTrigger>
          }
        />
        <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{children}</main>
      </div>
    </div>
  );
};

export default ChatShell;

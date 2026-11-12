"use client";

import React, { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
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
import { useTranslation } from "react-i18next";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sidebar,
  SidebarFooter,
  SidebarGroup,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  useSidebar,
} from "@/components/ui/sidebar";
import { MobileSidebarTrigger, useCloseMobileSidebarOn } from "@/components/shared/MobileSidebar";
import { navMenuButtonClassName, navMenuClassName, sidebarWidthStyle } from "@/components/shared/sidebarStyles";
import { SidebarBrand } from "@/components/shared/SidebarBrand";
import ChatAccountMenu from "./ChatAccountMenu";
import { Kbd } from "@/components/ui/kbd";
import { useShortcutLabel } from "@/components/CommandPalette/useShortcutLabel";
import { useHotkeys } from "react-hotkeys-hook";
import { DashboardHeader } from "@/components/DashboardHeader";
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
    <SidebarMenuItem>
      <SidebarMenuButton
        onClick={onClick}
        isActive={active}
        aria-label={label}
        tooltip={label}
        aria-current={active ? "page" : undefined}
        className={navMenuButtonClassName}
      >
        {icon}
        <span className="flex-1 truncate">{label}</span>
        {shortcut && (
          <Kbd className="bg-transparent text-muted-foreground group-data-[collapsible=icon]:hidden">{shortcut}</Kbd>
        )}
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

const ChatShell: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <SidebarProvider
    style={sidebarWidthStyle}
    className="chat-theme h-full min-h-0 overflow-hidden bg-background text-foreground"
  >
    <ChatShellLayout>{children}</ChatShellLayout>
  </SidebarProvider>
);

function ChatShellLayout({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation();
  const router = useRouter();
  const pathname = stripTrailingSlash(usePathname() ?? "");
  const { accessToken, conversations, activeConversationId, deleteConversation, renameConversation } = useChatShell();
  const handleLogout = useLogout(accessToken);
  const { state, isMobile, openMobile, setOpen, setOpenMobile } = useSidebar();
  const collapsed = state === "collapsed" && !isMobile;
  useCloseMobileSidebarOn();
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchAwaitingDrawer, setSearchAwaitingDrawer] = useState(false);
  const shortcut = useShortcutLabel();
  const openSearch = () => {
    if (isMobile && !openMobile) {
      setSearchAwaitingDrawer(true);
      setOpenMobile(true);
      return;
    }
    if (!isMobile) setOpen(true);
    setSearchOpen(true);
  };
  const openSearchOnceDrawerHasFocus = () => {
    if (!searchAwaitingDrawer) return;
    setSearchAwaitingDrawer(false);
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
    setOpenMobile(false);
    setSearchOpen(false);
    router.push(href);
  };

  const sidebar = (
    <>
      <SidebarBrand accessToken={accessToken} homeHref={routes.chats} homeLabel={t("chat.shell.home")} />
      <SidebarGroup className="gap-1 px-3 pt-3 pb-2">
        <ViewSwitcher
          collapsed={collapsed}
          className="h-9 w-full max-w-none justify-start shadow-none group-data-[collapsible=icon]:size-9 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:p-0 [&_[data-slot=workspace-label]]:flex-1 [&_[data-slot=workspace-label]]:text-left group-data-[collapsible=icon]:[&_[data-slot=workspace-label]]:hidden"
        />
        <SidebarMenu className={navMenuClassName}>
          <NavItem icon={<Plus />} label={t("chat.shell.newChat")} onClick={() => navigate(routes.chats)} />
        </SidebarMenu>
      </SidebarGroup>
      <nav aria-label={t("chat.shell.workspaceNav")} className="flex flex-none flex-col px-3 pb-3">
        <SidebarMenu className={navMenuClassName}>
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
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <SidebarMenuButton
                    isActive={workspaceActive}
                    aria-label={t("chat.shell.workspace")}
                    tooltip={t("chat.shell.workspace")}
                    aria-current={workspaceActive ? "page" : undefined}
                    className={navMenuButtonClassName}
                  />
                }
              >
                <Settings2 className="size-4" />
                <span className="flex-1 truncate text-left">{t("chat.shell.workspace")}</span>
                <ChevronDown className="size-3.5 group-data-[collapsible=icon]:hidden" />
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
          </SidebarMenuItem>
          <NavItem icon={<Search />} label={t("chat.shell.searchChats")} shortcut={shortcut} onClick={openSearch} />
        </SidebarMenu>
      </nav>
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden px-3 pb-3 group-data-[collapsible=icon]:hidden">
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
      <SidebarFooter className="mt-auto flex-none gap-2.5 border-t border-sidebar-border p-3">
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
    <>
      <Sidebar
        collapsible="icon"
        role="complementary"
        aria-label={t("chat.shell.chatNavigation")}
        className="border-sidebar-border"
      >
        <div className="contents" onFocus={openSearchOnceDrawerHasFocus}>
          {sidebar}
        </div>
      </Sidebar>
      <SidebarInset className="min-w-0 overflow-hidden">
        <DashboardHeader
          accessToken={accessToken}
          title={title}
          workspace="chat"
          navigationTrigger={<MobileSidebarTrigger label={t("chat.shell.openChatNavigation")} />}
        />
        <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{children}</div>
      </SidebarInset>
    </>
  );
}

export default ChatShell;

"use client";

import React from "react";
import { usePathname, useRouter } from "next/navigation";
import { Plus, MessageSquare, LayoutGrid, KeyRound, Lock, BarChart3, ScrollText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { migratedHref } from "@/utils/migratedPages";
import { useChatShell } from "@/contexts/ChatShellContext";
import ConversationList from "./ConversationList";
import { useTranslation } from "react-i18next";

export function getChatRoutes() {
  const base = migratedHref("chat");
  return {
    chats: base,
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
      className={`w-full justify-start gap-2.5 px-2.5 font-medium hover:bg-sidebar-accent ${
        active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-muted-foreground"
      }`}
    >
      <span className="shrink-0">{icon}</span>
      <span className="flex-1 text-left">{label}</span>
    </Button>
  );
}

interface ChatShellProps {
  children: React.ReactNode;
}

const ChatShell: React.FC<ChatShellProps> = ({ children }) => {
  const { t } = useTranslation("chat");
  const router = useRouter();
  const pathname = stripTrailingSlash(usePathname() ?? "");
  const { conversations, activeConversationId, deleteConversation, renameConversation } = useChatShell();

  const routes = getChatRoutes();
  const isChatsRoute = pathname === routes.chats;

  return (
    <div className="flex h-full w-full flex-col bg-background overflow-hidden">
      <div className="shrink-0 border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-center text-[13px] text-amber-800">
        {t("preview.warning")}{" "}
        <a
          href="https://github.com/BerriAI/litellm/discussions/32085"
          target="_blank"
          rel="noreferrer"
          className="font-medium underline"
        >
          {t("preview.link")}
        </a>
        .
      </div>
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <div className="shrink-0 bg-sidebar border-sidebar-border border-r flex flex-col overflow-hidden w-[260px]">
          <div className="px-2 pt-3 pb-1 shrink-0">
            <Button onClick={() => router.push(routes.chats)} className="w-full justify-start gap-2.5">
              <Plus className="h-4 w-4" />
              {t("navigation.newChat")}
            </Button>
          </div>

          <Separator className="mx-2 mt-2 shrink-0" />

          <div className="px-2 py-1 shrink-0">
            <NavItem
              icon={<MessageSquare className="h-4 w-4" />}
              label={t("navigation.chats")}
              onClick={() => router.push(routes.chats)}
              active={isChatsRoute}
            />
            <NavItem
              icon={<LayoutGrid className="h-4 w-4" />}
              label={t("navigation.integrations")}
              onClick={() => router.push(routes.integrations)}
              active={pathname === routes.integrations}
            />
            <NavItem
              icon={<KeyRound className="h-4 w-4" />}
              label={t("navigation.credentials")}
              onClick={() => router.push(routes.credentials)}
              active={pathname === routes.credentials}
            />
            <NavItem
              icon={<Lock className="h-4 w-4" />}
              label={t("navigation.apiKeys")}
              onClick={() => router.push(routes.apiKeys)}
              active={pathname === routes.apiKeys}
            />
            <NavItem
              icon={<ScrollText className="h-4 w-4" />}
              label={t("navigation.logs")}
              onClick={() => router.push(routes.logs)}
              active={pathname === routes.logs}
            />
            <NavItem
              icon={<BarChart3 className="h-4 w-4" />}
              label={t("navigation.usage")}
              onClick={() => router.push(routes.usage)}
              active={pathname === routes.usage}
            />
          </div>

          <Separator className="mx-2 shrink-0" />

          <div className="flex-1 overflow-hidden flex flex-col">
            <ConversationList
              conversations={conversations}
              activeConversationId={activeConversationId}
              onSelect={(id) => router.push(`${routes.chats}?id=${id}`)}
              onDelete={(id) => {
                deleteConversation(id);
                if (id === activeConversationId) router.push(routes.chats);
              }}
              onRename={renameConversation}
            />
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden min-w-0">{children}</div>
      </div>
    </div>
  );
};

export default ChatShell;

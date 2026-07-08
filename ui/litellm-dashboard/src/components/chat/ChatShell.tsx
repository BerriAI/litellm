"use client";

import React from "react";
import { usePathname, useRouter } from "next/navigation";
import { Plus, MessageSquare, LayoutGrid, KeyRound, Lock, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
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
  const router = useRouter();
  const pathname = stripTrailingSlash(usePathname() ?? "");
  const { conversations, activeConversationId, deleteConversation, renameConversation } = useChatShell();

  const isChatsRoute = pathname === CHAT_ROUTES.chats;

  return (
    <div className="flex h-full w-full flex-col bg-background overflow-hidden">
      <div className="shrink-0 border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-center text-[13px] text-amber-800">
        This is a pre-v0 feature. Do not use in production, it may change unexpectedly. Please share feedback{" "}
        <a
          href="https://github.com/BerriAI/litellm/discussions/32085"
          target="_blank"
          rel="noreferrer"
          className="font-medium underline"
        >
          here
        </a>
        .
      </div>
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <div className="shrink-0 bg-sidebar border-sidebar-border border-r flex flex-col overflow-hidden w-[260px]">
          <div className="px-2 pt-3 pb-1 shrink-0">
            <Button onClick={() => router.push(CHAT_ROUTES.chats)} className="w-full justify-start gap-2.5">
              <Plus className="h-4 w-4" />
              New Chat
            </Button>
          </div>

          <Separator className="mx-2 mt-2 shrink-0" />

          <div className="px-2 py-1 shrink-0">
            <NavItem
              icon={<MessageSquare className="h-4 w-4" />}
              label="Chats"
              onClick={() => router.push(CHAT_ROUTES.chats)}
              active={isChatsRoute}
            />
            <NavItem
              icon={<LayoutGrid className="h-4 w-4" />}
              label="Integrations"
              onClick={() => router.push(CHAT_ROUTES.integrations)}
              active={pathname === CHAT_ROUTES.integrations}
            />
            <NavItem
              icon={<KeyRound className="h-4 w-4" />}
              label="Credentials"
              onClick={() => router.push(CHAT_ROUTES.credentials)}
              active={pathname === CHAT_ROUTES.credentials}
            />
            <NavItem
              icon={<Lock className="h-4 w-4" />}
              label="API Keys"
              onClick={() => router.push(CHAT_ROUTES.apiKeys)}
              active={pathname === CHAT_ROUTES.apiKeys}
            />
            <NavItem
              icon={<BarChart3 className="h-4 w-4" />}
              label="Usage"
              onClick={() => router.push(CHAT_ROUTES.usage)}
              active={pathname === CHAT_ROUTES.usage}
            />
          </div>

          <Separator className="mx-2 shrink-0" />

          <div className="flex-1 overflow-hidden flex flex-col">
            <ConversationList
              conversations={conversations}
              activeConversationId={activeConversationId}
              onSelect={(id) => router.push(`${CHAT_ROUTES.chats}?id=${id}`)}
              onDelete={(id) => {
                deleteConversation(id);
                if (id === activeConversationId) router.push(CHAT_ROUTES.chats);
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

"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Pencil, Trash2, Search, MessageSquare, Clock3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import dayjs from "dayjs";
import { Conversation } from "./types";

interface Props {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, newTitle: string) => void;
  searchOpen?: boolean;
  onSearchOpenChange?: (open: boolean) => void;
  hideSearchButton?: boolean;
}

type DateGroup = "Recents" | "Yesterday" | "Last 7 Days" | "Older";

const getDateGroup = (timestamp: number): DateGroup => {
  const now = dayjs();
  const date = dayjs(timestamp);
  if (date.isSame(now, "day")) return "Recents";
  if (date.isSame(now.subtract(1, "day"), "day")) return "Yesterday";
  if (date.isAfter(now.subtract(7, "day"))) return "Last 7 Days";
  return "Older";
};

const DATE_GROUP_ORDER: DateGroup[] = ["Recents", "Yesterday", "Last 7 Days", "Older"];

interface GroupedConversations {
  group: DateGroup;
  items: Conversation[];
}

const groupConversations = (conversations: Conversation[]): GroupedConversations[] => {
  const map = new Map<DateGroup, Conversation[]>();
  for (const conv of conversations) {
    const group = getDateGroup(conv.updatedAt);
    if (!map.has(group)) map.set(group, []);
    map.get(group)!.push(conv);
  }
  return DATE_GROUP_ORDER.filter((g) => map.has(g)).map((g) => ({
    group: g,
    items: map.get(g)!,
  }));
};

interface ConversationRowProps {
  conv: Conversation;
  isActive: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, newTitle: string) => void;
}

const ConversationRow: React.FC<ConversationRowProps> = ({ conv, isActive, onSelect, onDelete, onRename }) => {
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(conv.title);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const startEditing = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditValue(conv.title);
    setEditing(true);
  };

  const commitRename = () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== conv.title) {
      onRename(conv.id, trimmed);
    }
    setEditing(false);
  };

  const cancelEditing = () => {
    setEditValue(conv.title);
    setEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitRename();
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancelEditing();
    }
  };

  return (
    <div
      className={`chat-conversation-row group flex items-center rounded-lg transition-colors min-h-9 relative ${
        isActive ? "bg-sidebar-accent text-sidebar-accent-foreground" : "hover:bg-sidebar-accent"
      }`}
    >
      {editing ? (
        <Input
          ref={inputRef}
          aria-label="Conversation title"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={commitRename}
          onClick={(e) => e.stopPropagation()}
          className="h-7 text-[13px] flex-1"
        />
      ) : (
        <>
          <Button
            variant="ghost"
            onClick={() => onSelect(conv.id)}
            aria-label={conv.title}
            aria-current={isActive ? "page" : undefined}
            className={`min-w-0 flex-1 justify-start text-[13px] hover:bg-sidebar-accent ${
              isActive ? "font-medium" : ""
            }`}
            title={conv.title}
          >
            <span className="min-w-0 flex-1 text-left">
              <span className="block truncate">{conv.title}</span>
              <span className="moyai-chat-conversation-meta block truncate text-xs font-normal text-muted-foreground">
                {conv.model} · {dayjs(conv.updatedAt).format("MMM D")}
              </span>
            </span>
          </Button>

          <div
            className="chat-conversation-actions flex gap-0.5 pr-1 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity shrink-0"
            onClick={(e) => e.stopPropagation()}
          >
            <TooltipProvider delay={300}>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      onClick={startEditing}
                      variant="ghost"
                      size="icon-xs"
                      aria-label={`Rename ${conv.title}`}
                      className="text-muted-foreground"
                    >
                      <Pencil className="h-3 w-3" />
                    </Button>
                  }
                />
                <TooltipContent side="bottom">
                  <p>Rename</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <AlertDialog>
              <TooltipProvider delay={300}>
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <AlertDialogTrigger
                        render={
                          <Button
                            variant="ghost"
                            size="icon-xs"
                            aria-label={`Delete ${conv.title}`}
                            className="text-muted-foreground hover:text-destructive"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        }
                      />
                    }
                  />
                  <TooltipContent side="bottom">
                    <p>Delete</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <AlertDialogContent className="chat-theme">
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
                  <AlertDialogDescription>This action cannot be undone</AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => onDelete(conv.id)}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </>
      )}
    </div>
  );
};

interface SearchModalProps {
  open: boolean;
  conversations: Conversation[];
  onSelect: (id: string) => void;
  onClose: () => void;
}

const SearchModal: React.FC<SearchModalProps> = ({ open, conversations, onSelect, onClose }) => {
  const [query, setQuery] = useState("");
  const [wasOpen, setWasOpen] = useState(open);

  if (open !== wasOpen) {
    setWasOpen(open);
    if (!open) setQuery("");
  }

  const filtered = query.trim()
    ? conversations.filter((c) => c.title.toLowerCase().includes(query.trim().toLowerCase()))
    : conversations;

  const handleSelect = (id: string) => {
    onSelect(id);
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="chat-theme sm:max-w-[480px] p-4 gap-0">
        <DialogTitle className="sr-only">Search conversations</DialogTitle>
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            autoFocus
            aria-label="Search conversations"
            placeholder="Search conversations\u2026"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        <ScrollArea className="h-[320px]">
          {filtered.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground text-sm">No conversations found</div>
          ) : (
            filtered.map((conv) => {
              return (
                <Button
                  key={conv.id}
                  variant="ghost"
                  onClick={() => handleSelect(conv.id)}
                  className="w-full justify-start gap-2 px-2.5"
                >
                  <MessageSquare className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="text-left text-[13px] flex-1 truncate">{conv.title}</span>
                  <span className="text-[11px] text-muted-foreground shrink-0 ml-auto">
                    {dayjs(conv.updatedAt).format("MMM D")}
                  </span>
                </Button>
              );
            })
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

const ConversationList: React.FC<Props> = ({
  conversations,
  activeConversationId,
  onSelect,
  onDelete,
  onRename,
  searchOpen,
  onSearchOpenChange,
  hideSearchButton = false,
}) => {
  const [localSearchOpen, setLocalSearchOpen] = useState(false);
  const searchModalOpen = searchOpen ?? localSearchOpen;
  const setSearchModalOpen = onSearchOpenChange ?? setLocalSearchOpen;

  const handleGlobalKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSearchModalOpen(!searchModalOpen);
      }
    },
    [searchModalOpen, setSearchModalOpen],
  );

  useEffect(() => {
    document.addEventListener("keydown", handleGlobalKeyDown);
    return () => document.removeEventListener("keydown", handleGlobalKeyDown);
  }, [handleGlobalKeyDown]);

  const grouped = groupConversations(conversations);

  return (
    <>
      <div className="flex flex-col h-full w-full overflow-hidden">
        {!hideSearchButton && (
          <div className="px-2 pt-2">
            <Button
              variant="ghost"
              onClick={() => setSearchModalOpen(true)}
              className="chat-nav-item w-full justify-start gap-2.5 text-muted-foreground hover:bg-sidebar-accent"
            >
              <Search className="size-4" />
              Search chats
            </Button>
          </div>
        )}
        <ScrollArea className="flex-1 h-0 px-1.5 pt-2">
          {grouped.length === 0 ? (
            <div>
              <div className="moyai-chat-history-heading flex items-center gap-2 text-xs text-sidebar-foreground">
                <Clock3 className="size-3.5" aria-hidden="true" />
                <span className="flex-1">Recent</span>
                <span>0</span>
              </div>
              <p className="moyai-chat-empty-history text-xs text-muted-foreground">No chats here yet.</p>
            </div>
          ) : (
            grouped.map(({ group, items }) => (
              <div key={group} className="mb-2">
                <div className="moyai-chat-history-heading flex items-center gap-2 px-2 pt-2 pb-1 text-xs font-normal text-sidebar-foreground">
                  <Clock3 className="size-3.5" aria-hidden="true" />
                  {group === "Recents" ? "Recent" : group}
                </div>
                {items.map((conv) => (
                  <ConversationRow
                    key={conv.id}
                    conv={conv}
                    isActive={conv.id === activeConversationId}
                    onSelect={onSelect}
                    onDelete={onDelete}
                    onRename={onRename}
                  />
                ))}
              </div>
            ))
          )}
        </ScrollArea>
      </div>

      <SearchModal
        open={searchModalOpen}
        conversations={conversations}
        onSelect={onSelect}
        onClose={() => setSearchModalOpen(false)}
      />
    </>
  );
};

export default ConversationList;

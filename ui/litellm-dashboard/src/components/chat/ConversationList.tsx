"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Pencil, Trash2, Search, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent } from "@/components/ui/dialog";
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

  const truncatedTitle = conv.title.length > 40 ? conv.title.slice(0, 40) + "\u2026" : conv.title;

  return (
    <div
      onClick={() => !editing && onSelect(conv.id)}
      className={`group flex items-center px-2 py-1.5 rounded-md cursor-pointer transition-colors min-h-[34px] relative ${
        isActive ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
      }`}
    >
      {editing ? (
        <Input
          ref={inputRef}
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={commitRename}
          onClick={(e) => e.stopPropagation()}
          className="h-7 text-[13px] flex-1"
        />
      ) : (
        <>
          <span
            className={`flex-1 text-[13px] overflow-hidden whitespace-nowrap text-ellipsis ${
              isActive ? "font-medium" : ""
            }`}
            title={conv.title}
          >
            {truncatedTitle}
          </span>

          <div
            className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
            onClick={(e) => e.stopPropagation()}
          >
            <TooltipProvider delay={300}>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button onClick={startEditing} variant="ghost" size="icon-xs" className="text-muted-foreground">
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
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
                  <AlertDialogDescription>This action cannot be undone</AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => onDelete(conv.id)}
                    className="bg-destructive text-white hover:bg-destructive/90"
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
      <DialogContent className="sm:max-w-[480px] p-4 gap-0">
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            autoFocus
            placeholder="Search conversations\u2026"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        <ScrollArea className="max-h-[320px]">
          {filtered.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground text-sm">No conversations found</div>
          ) : (
            filtered.map((conv) => {
              const truncated = conv.title.length > 55 ? conv.title.slice(0, 55) + "\u2026" : conv.title;
              return (
                <div
                  key={conv.id}
                  onClick={() => handleSelect(conv.id)}
                  className="flex items-center gap-2 px-2.5 py-2 rounded-md cursor-pointer transition-colors hover:bg-accent/50"
                >
                  <MessageSquare className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="text-[13px] flex-1 truncate">{truncated}</span>
                  <span className="text-[11px] text-muted-foreground shrink-0 ml-auto">
                    {dayjs(conv.updatedAt).format("MMM D")}
                  </span>
                </div>
              );
            })
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

const ConversationList: React.FC<Props> = ({ conversations, activeConversationId, onSelect, onDelete, onRename }) => {
  const [searchModalOpen, setSearchModalOpen] = useState(false);

  const handleGlobalKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      setSearchModalOpen((prev) => !prev);
    }
  }, []);

  useEffect(() => {
    document.addEventListener("keydown", handleGlobalKeyDown);
    return () => document.removeEventListener("keydown", handleGlobalKeyDown);
  }, [handleGlobalKeyDown]);

  const grouped = groupConversations(conversations);

  return (
    <>
      <div className="flex flex-col h-full w-full overflow-hidden">
        <ScrollArea className="flex-1 h-0 px-1.5 pt-2">
          {grouped.length === 0 ? (
            <div className="text-center text-muted-foreground/60 text-xs mt-8 px-3">
              No conversations yet
              <br />
              Start a new chat above
            </div>
          ) : (
            grouped.map(({ group, items }) => (
              <div key={group} className="mb-2">
                <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider px-2 pt-2 pb-1">
                  {group}
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

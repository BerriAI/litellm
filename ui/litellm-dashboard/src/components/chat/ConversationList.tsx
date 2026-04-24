"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Pencil as EditOutlined,
  Trash2 as DeleteOutlined,
  Plus as PlusOutlined,
  Search as SearchOutlined,
  User as UserOutlined,
  MessageSquare as MessageOutlined,
} from "lucide-react";
import dayjs from "dayjs";
import { Conversation } from "./types";

const Text = ({
  className,
  style,
  children,
  title,
}: {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
  title?: string;
}) => (
  <span className={className} style={style} title={title}>
    {children}
  </span>
);

interface Props {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNewChat: () => void;
  onRename: (id: string, newTitle: string) => void;
}

// ---- Date grouping helpers ----

type DateGroup = "Today" | "Yesterday" | "Last 7 Days" | "Older";

const getDateGroup = (timestamp: number): DateGroup => {
  const now = dayjs();
  const date = dayjs(timestamp);

  if (date.isSame(now, "day")) return "Today";
  if (date.isSame(now.subtract(1, "day"), "day")) return "Yesterday";
  if (date.isAfter(now.subtract(7, "day"))) return "Last 7 Days";
  return "Older";
};

const DATE_GROUP_ORDER: DateGroup[] = ["Today", "Yesterday", "Last 7 Days", "Older"];

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

// ---- Delete confirmation popover (replaces antd Popconfirm) ----

const DeleteConfirmPopover: React.FC<{ onConfirm: () => void }> = ({
  onConfirm,
}) => {
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          aria-label="Delete"
          className="h-[22px] w-[22px] p-0 min-w-[22px] text-destructive hover:text-destructive"
          onClick={(e) => e.stopPropagation()}
        >
          <DeleteOutlined className="h-3 w-3" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-56"
        align="end"
        onClick={(e) => e.stopPropagation()}
      >
        <p className="text-sm mb-3">Delete this conversation?</p>
        <div className="flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => {
              setOpen(false);
              onConfirm();
            }}
          >
            Delete
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
};

// ---- Single conversation row ----

interface ConversationRowProps {
  conv: Conversation;
  isActive: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, newTitle: string) => void;
}

const ConversationRow: React.FC<ConversationRowProps> = ({
  conv,
  isActive,
  onSelect,
  onDelete,
  onRename,
}) => {
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

  const truncatedTitle =
    conv.title.length > 40 ? conv.title.slice(0, 40) + "…" : conv.title;

  return (
    <div
      onClick={() => !editing && onSelect(conv.id)}
      className="conversation-row group"
      style={{
        display: "flex",
        alignItems: "center",
        padding: "6px 8px",
        borderRadius: 6,
        cursor: editing ? "default" : "pointer",
        backgroundColor: isActive ? "#e6f4ff" : "transparent",
        transition: "background-color 0.15s",
        minHeight: 34,
        position: "relative",
      }}
      onMouseEnter={(e) => {
        if (!isActive) {
          (e.currentTarget as HTMLDivElement).style.backgroundColor = "#f5f5f5";
        }
      }}
      onMouseLeave={(e) => {
        if (!isActive) {
          (e.currentTarget as HTMLDivElement).style.backgroundColor = "transparent";
        }
      }}
    >
      {editing ? (
        <Input
          ref={inputRef}
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={commitRename}
          onClick={(e) => e.stopPropagation()}
          className="flex-1 h-7 text-[13px]"
        />
      ) : (
        <>
          <Text
            style={{
              flex: 1,
              fontSize: 13,
              color: isActive ? "#1677ff" : "#333",
              overflow: "hidden",
              whiteSpace: "nowrap",
              textOverflow: "ellipsis",
              fontWeight: isActive ? 500 : 400,
            }}
            title={conv.title}
          >
            {truncatedTitle}
          </Text>

          {/* Action icons — visible only on hover via CSS opacity */}
          <div
            className="conversation-actions"
            style={{
              display: "flex",
              gap: 2,
              opacity: 0,
              transition: "opacity 0.15s",
              flexShrink: 0,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={startEditing}
                    aria-label="Rename"
                    className="h-[22px] w-[22px] p-0 min-w-[22px]"
                  >
                    <EditOutlined className="h-3 w-3" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Rename</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <DeleteConfirmPopover onConfirm={() => onDelete(conv.id)} />
          </div>
        </>
      )}
    </div>
  );
};

// ---- Cmd+K search modal ----

interface SearchModalProps {
  open: boolean;
  conversations: Conversation[];
  onSelect: (id: string) => void;
  onClose: () => void;
}

const SearchModal: React.FC<SearchModalProps> = ({
  open,
  conversations,
  onSelect,
  onClose,
}) => {
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const filtered = query.trim()
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(query.trim().toLowerCase())
      )
    : conversations;

  const handleSelect = (id: string) => {
    onSelect(id);
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(o) => (!o ? onClose() : undefined)}>
      <DialogContent className="sm:max-w-[480px] p-4">
        <DialogHeader className="sr-only">
          <DialogTitle>Search conversations</DialogTitle>
        </DialogHeader>
        <div className="relative mb-3">
          <SearchOutlined className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            autoFocus
            placeholder="Search conversations…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        <div style={{ maxHeight: 320, overflowY: "auto" }}>
          {filtered.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground">
              No conversations found
            </div>
          ) : (
            filtered.map((conv) => {
              const truncated =
                conv.title.length > 55
                  ? conv.title.slice(0, 55) + "…"
                  : conv.title;
              return (
                <div
                  key={conv.id}
                  onClick={() => handleSelect(conv.id)}
                  className="flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer hover:bg-accent"
                >
                  <MessageOutlined className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <Text style={{ fontSize: 13 }}>{truncated}</Text>
                  <span className="ml-auto flex-shrink-0 text-xs text-muted-foreground">
                    {dayjs(conv.updatedAt).format("MMM D")}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

// ---- Main ConversationList component ----

const ConversationList: React.FC<Props> = ({
  conversations,
  activeConversationId,
  onSelect,
  onDelete,
  onNewChat,
  onRename,
}) => {
  const [searchModalOpen, setSearchModalOpen] = useState(false);

  // Cmd+K / Ctrl+K listener
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
      {/* Hover-reveal CSS for action icons */}
      <style>{`
        .conversation-row:hover .conversation-actions {
          opacity: 1 !important;
        }
      `}</style>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          height: "100%",
          width: "100%",
          overflow: "hidden",
        }}
      >
        {/* Top: New Chat button */}
        <div style={{ padding: "12px 10px 8px" }}>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={onNewChat}
                  className="w-full"
                >
                  <PlusOutlined className="mr-2 h-4 w-4" />
                  New Chat
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                Chats are saved locally in this browser. All requests are
                logged in Spend → Logs.
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        {/* Conversation list (scrollable) */}
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "0 6px",
          }}
        >
          {grouped.length === 0 ? (
            <div
              style={{
                textAlign: "center",
                color: "#bbb",
                fontSize: 12,
                marginTop: 32,
                padding: "0 12px",
              }}
            >
              No conversations yet.
              <br />
              Start a new chat above.
            </div>
          ) : (
            grouped.map(({ group, items }) => (
              <div key={group} style={{ marginBottom: 8 }}>
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#999",
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                    padding: "8px 8px 4px",
                  }}
                >
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
        </div>

        {/* Bottom: user avatar placeholder */}
        <div
          style={{
            padding: "10px 12px",
            borderTop: "1px solid #f0f0f0",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <Avatar className="h-7 w-7 flex-shrink-0">
            <AvatarFallback
              // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
              className="bg-indigo-100 text-indigo-600"
            >
              <UserOutlined className="h-3.5 w-3.5" />
            </AvatarFallback>
          </Avatar>
          <Text
            style={{
              fontSize: 13,
              color: "#555",
              overflow: "hidden",
              whiteSpace: "nowrap",
              textOverflow: "ellipsis",
            }}
          >
            My Account
          </Text>
        </div>
      </div>

      {/* Cmd+K search modal */}
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

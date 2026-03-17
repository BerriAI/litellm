"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Button,
  Input,
  Modal,
  Popconfirm,
  Tooltip,
  Avatar,
  Typography,
} from "antd";
import {
  EditOutlined,
  DeleteOutlined,
  PlusOutlined,
  SearchOutlined,
  UserOutlined,
  MessageOutlined,
} from "@ant-design/icons";
import dayjs from "dayjs";
import { Conversation } from "./types";

const { Text } = Typography;

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
          ref={(node) => {
            inputRef.current = node?.input ?? null;
          }}
          size="small"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={commitRename}
          onClick={(e) => e.stopPropagation()}
          style={{ flex: 1, fontSize: 13 }}
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
            <Tooltip title="Rename">
              <Button
                type="text"
                size="small"
                icon={<EditOutlined style={{ fontSize: 12 }} />}
                onClick={startEditing}
                style={{ width: 22, height: 22, padding: 0, minWidth: 22 }}
              />
            </Tooltip>
            <Popconfirm
              title="Delete this conversation?"
              onConfirm={() => onDelete(conv.id)}
              okText="Delete"
              cancelText="Cancel"
              okButtonProps={{ danger: true }}
            >
              <Tooltip title="Delete">
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined style={{ fontSize: 12 }} />}
                  style={{ width: 22, height: 22, padding: 0, minWidth: 22 }}
                />
              </Tooltip>
            </Popconfirm>
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
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={null}
      width={480}
      styles={{ body: { padding: "16px 16px 8px" } }}
    >
      <Input
        autoFocus
        prefix={<SearchOutlined style={{ color: "#bbb" }} />}
        placeholder="Search conversations…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ marginBottom: 12 }}
        allowClear
      />

      <div style={{ maxHeight: 320, overflowY: "auto" }}>
        {filtered.length === 0 ? (
          <div style={{ textAlign: "center", padding: "24px 0", color: "#999" }}>
            No conversations found
          </div>
        ) : (
          filtered.map((conv) => {
            const truncated =
              conv.title.length > 55 ? conv.title.slice(0, 55) + "…" : conv.title;
            return (
              <div
                key={conv.id}
                onClick={() => handleSelect(conv.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "8px 10px",
                  borderRadius: 6,
                  cursor: "pointer",
                  transition: "background-color 0.1s",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLDivElement).style.backgroundColor = "#f0f5ff";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLDivElement).style.backgroundColor = "transparent";
                }}
              >
                <MessageOutlined style={{ color: "#999", flexShrink: 0 }} />
                <Text style={{ fontSize: 13 }}>{truncated}</Text>
                <Text
                  type="secondary"
                  style={{ fontSize: 11, marginLeft: "auto", flexShrink: 0 }}
                >
                  {dayjs(conv.updatedAt).format("MMM D")}
                </Text>
              </div>
            );
          })
        )}
      </div>
    </Modal>
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
          <Tooltip
            title="Chats are saved locally in this browser. All requests are logged in Spend → Logs."
            placement="right"
          >
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={onNewChat}
              style={{ width: "100%" }}
            >
              New Chat
            </Button>
          </Tooltip>
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
          <Avatar
            size={28}
            icon={<UserOutlined />}
            style={{ backgroundColor: "#e0e7ff", color: "#4f46e5", flexShrink: 0 }}
          />
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

"use client";

import React, { useState } from "react";
import { MessageSquare, Search } from "lucide-react";
import dayjs from "dayjs";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Conversation } from "./types";

interface ConversationSearchDialogProps {
  open: boolean;
  conversations: Conversation[];
  onSelect: (id: string) => void;
  onClose: () => void;
}

const ConversationSearchDialog: React.FC<ConversationSearchDialogProps> = ({
  open,
  conversations,
  onSelect,
  onClose,
}) => {
  const { t } = useTranslation();
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
        <DialogTitle className="sr-only">{t("chat.conversationList.searchTitle")}</DialogTitle>
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            autoFocus
            aria-label={t("chat.conversationList.searchLabel")}
            placeholder={t("chat.conversationList.searchPlaceholder")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        <ScrollArea className="h-[320px]">
          {filtered.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground text-sm">{t("chat.conversationList.noResults")}</div>
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

export default ConversationSearchDialog;

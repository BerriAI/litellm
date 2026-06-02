"use client";

/**
 * Composer — textarea + send button at the bottom of the conversation pane.
 *
 * On submit, POSTs to the followup endpoint and lets the SSE stream
 * surface the resulting `user_message` event via the parent's event list.
 */
import { useState } from "react";
import { Button, Input, message } from "antd";
import { sendSessionFollowup } from "@/lib/cloud-agents-client";

interface ComposerProps {
  sessionId: string;
  accessToken: string | null;
  onSent?: () => void;
}

export default function Composer({ sessionId, accessToken, onSent }: ComposerProps) {
  const [value, setValue] = useState("");
  const [sending, setSending] = useState(false);

  const handleSend = async () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    setSending(true);
    try {
      await sendSessionFollowup(accessToken, sessionId, trimmed);
      setValue("");
      onSent?.();
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);
      message.error(`Send failed: ${errMsg}`);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="border-t border-gray-200 p-3" data-testid="composer">
      <div className="flex items-end gap-2">
        <Input.TextArea
          rows={2}
          autoSize={{ minRows: 2, maxRows: 6 }}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Add a follow-up..."
          disabled={sending}
          data-testid="composer-input"
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault();
              void handleSend();
            }
          }}
        />
        <Button
          type="primary"
          loading={sending}
          onClick={() => void handleSend()}
          data-testid="composer-send"
          disabled={!value.trim()}
        >
          Send
        </Button>
      </div>
    </div>
  );
}

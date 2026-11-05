"use client";

import { useId, useState } from "react";
import { Check, ExternalLink, KeyRound } from "lucide-react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { InputGroupButton } from "@/components/ui/input-group";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import type { MCPServer } from "@/components/mcp_tools/types";
import { fetchClient } from "@/lib/http/api";
import { ApiError } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import i18n from "@/i18n/i18n";

const saveErrorMessage = (error: unknown, t: TFunction = i18n.t.bind(i18n)): string => {
  if (error instanceof ApiError) {
    const detail = (error.body as { detail?: { error?: string } } | null)?.detail?.error;
    if (detail) return detail;
  }
  return error instanceof Error && error.message ? error.message : t("playground.chat.byok.failed");
};

export function ByokConnect({
  server,
  name,
  onConnected,
  onCancel,
}: {
  server: MCPServer;
  name: string;
  onConnected: () => void;
  onCancel?: () => void;
}) {
  const { t } = useTranslation();
  const [credential, setCredential] = useState("");
  const [save, setSave] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const keyId = useId();
  const saveId = useId();

  const connect = async () => {
    if (!credential.trim()) {
      toast.error(t("playground.chat.byok.enterKey"));
      return;
    }
    setIsSaving(true);
    try {
      await fetchClient.POST("/v1/mcp/server/{server_id}/user-credential", {
        params: { path: { server_id: server.server_id } },
        body: { credential: credential.trim(), save },
      });
      toast.success(t("playground.chat.byok.connected", { name }));
      setCredential("");
      onConnected();
    } catch (error) {
      toast.error(saveErrorMessage(error, t));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <FieldGroup className="gap-4 p-4">
      <div className="flex flex-col gap-1">
        <p className="flex items-center gap-2 text-sm font-medium">
          <KeyRound className="size-4 text-muted-foreground" />
          {server.has_user_credential
            ? t("playground.chat.byok.replaceTitle", { name })
            : t("playground.chat.byok.connectTitle", { name })}
        </p>
        <p className="text-sm text-muted-foreground">
          {server.has_user_credential
            ? t("playground.chat.byok.replaceDescription", { name })
            : t("playground.chat.byok.connectDescription", { name })}
        </p>
      </div>
      {server.byok_description && server.byok_description.length > 0 && (
        <ul className="flex flex-col gap-1.5 text-sm">
          {server.byok_description.map((item) => (
            <li key={item} className="flex items-center gap-2">
              <Check className="size-3.5 shrink-0 text-muted-foreground" />
              {item}
            </li>
          ))}
        </ul>
      )}
      <Field>
        <FieldLabel htmlFor={keyId}>{t("playground.chat.byok.keyLabel", { name })}</FieldLabel>
        <PasswordInput
          id={keyId}
          value={credential}
          onChange={(event) => setCredential(event.target.value)}
          placeholder={t("playground.chat.byok.keyPlaceholder")}
        />
        {server.byok_api_key_help_url && (
          <FieldDescription>
            <a href={server.byok_api_key_help_url} target="_blank" rel="noopener noreferrer">
              {t("playground.chat.byok.helpLink")}
              <ExternalLink className="ml-1 inline size-3" />
            </a>
          </FieldDescription>
        )}
      </Field>
      <Field orientation="horizontal">
        <FieldLabel htmlFor={saveId} className="font-normal">
          {t("playground.chat.byok.save")}
        </FieldLabel>
        <Switch id={saveId} checked={save} onCheckedChange={setSave} />
      </Field>
      <div className="flex gap-2">
        <Button size="sm" onClick={() => void connect()} disabled={isSaving}>
          {isSaving && <Spinner />}
          {t("playground.chat.byok.connect")}
        </Button>
        {onCancel && (
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={isSaving}>
            {t("playground.chat.byok.cancel")}
          </Button>
        )}
      </div>
    </FieldGroup>
  );
}

export function ByokKeyButton({
  server,
  name,
  onConnected,
}: {
  server: MCPServer;
  name: string;
  onConnected: () => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger render={<InputGroupButton size="xs" className="shrink-0" />}>
        <KeyRound />
        {server.has_user_credential ? t("playground.chat.byok.replaceKey") : t("playground.chat.byok.connectKey")}
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 p-0">
        <ByokConnect
          server={server}
          name={name}
          onConnected={() => {
            setOpen(false);
            onConnected();
          }}
        />
      </PopoverContent>
    </Popover>
  );
}

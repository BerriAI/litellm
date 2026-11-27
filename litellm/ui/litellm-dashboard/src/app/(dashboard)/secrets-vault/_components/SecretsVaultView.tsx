"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Copy, Eye, EyeOff, KeyRound, Pencil, Plus, Trash2 } from "lucide-react";
import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";

import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { PageContent } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { $api, fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import { copyToClipboard } from "@/utils/dataUtils";

import { SecretModal } from "./SecretModal";
import { toSecretWrite, type SecretFormValues } from "./secretForm";

type Secret = components["schemas"]["VaultSecretMetadata"];

const LIST_KEY = ["get", "/vault/secrets"] as const;
const MASK = "••••••••••••";

const formatTime = (iso: string | null | undefined): string => (iso ? new Date(iso).toLocaleString() : "-");

const code = (
  <code className="rounded-sm border border-border bg-muted px-1 py-0.5 font-mono text-xs text-foreground" />
);

export const SecretsVaultView: React.FC = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [revealed, setRevealed] = useState<Readonly<Record<string, string>>>({});
  const [modal, setModal] = useState<{ editing?: Secret } | null>(null);
  const [deleting, setDeleting] = useState<Secret | null>(null);

  const { data, isLoading } = $api.useQuery("get", "/vault/secrets");
  const secrets = data?.secrets ?? [];

  const refresh = () => queryClient.invalidateQueries({ queryKey: LIST_KEY });
  const hide = (name: string) => setRevealed(({ [name]: _hidden, ...rest }) => rest);
  const onError = (error: unknown) => toast.error(extractProxyErrorMessage(error));

  const save = $api.useMutation("put", "/vault/secrets/{secret_name}", {
    onSuccess: (secret) => {
      toast.success(t("secretsVault.toasts.saved", { name: secret.secret_name }));
      hide(secret.secret_name);
      setModal(null);
      refresh();
    },
    onError,
  });

  const remove = $api.useMutation("delete", "/vault/secrets/{secret_name}", {
    onSuccess: (result) => {
      toast.success(t("secretsVault.toasts.deleted", { name: result.secret_name }));
      hide(result.secret_name);
      setDeleting(null);
      refresh();
    },
    onError,
  });

  const fetchValue = async (name: string): Promise<string | null> => {
    try {
      const response = await fetchClient.GET("/vault/secrets/{secret_name}", {
        params: { path: { secret_name: name } },
      });
      refresh();
      return response.data?.value ?? null;
    } catch (error) {
      onError(error);
      return null;
    }
  };

  const toggleReveal = async (name: string) => {
    if (name in revealed) return hide(name);
    const value = await fetchValue(name);
    if (value !== null) setRevealed((current) => ({ ...current, [name]: value }));
  };

  const copy = async (name: string) => {
    const value = revealed[name] ?? (await fetchValue(name));
    if (value !== null) await copyToClipboard(value, t("secretsVault.toasts.copied", { name }));
  };

  const handleSave = (values: SecretFormValues) =>
    save.mutate({ params: { path: { secret_name: values.name } }, body: toSecretWrite(values) });

  return (
    <>
      <PageContent>
        <PageHeader>
          <PageHeaderTitle>
            <KeyRound />
            {t("secretsVault.title")}
            <Badge variant="secondary">{t("secretsVault.experimental")}</Badge>
          </PageHeaderTitle>
          <PageHeaderDescription>{t("secretsVault.intro")}</PageHeaderDescription>
          <PageHeaderControls>
            <Button onClick={() => setModal({})}>
              <Plus />
              {t("secretsVault.newSecret")}
            </Button>
          </PageHeaderControls>
        </PageHeader>

        <Alert variant="info" className="mb-4">
          <AlertTitle>{t("secretsVault.usage.title")}</AlertTitle>
          <AlertDescription>
            <Trans i18nKey="secretsVault.usage.description" components={{ code }} />
          </AlertDescription>
        </Alert>

        {!isLoading && secrets.length === 0 ? (
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <KeyRound />
              </EmptyMedia>
              <EmptyTitle>{t("secretsVault.empty.title")}</EmptyTitle>
              <EmptyDescription>{t("secretsVault.empty.description")}</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("secretsVault.columns.name")}</TableHead>
                <TableHead>{t("secretsVault.columns.value")}</TableHead>
                <TableHead>{t("secretsVault.columns.description")}</TableHead>
                <TableHead>{t("secretsVault.columns.updated")}</TableHead>
                <TableHead>{t("secretsVault.columns.lastAccessed")}</TableHead>
                <TableHead className="text-right">{t("secretsVault.columns.actions")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {secrets.map((secret) => {
                const name = secret.secret_name;
                const isRevealed = name in revealed;
                return (
                  <TableRow key={name}>
                    <TableCell className="font-mono font-medium">{name}</TableCell>
                    <TableCell className="max-w-[320px] break-all font-mono text-xs">
                      {isRevealed ? revealed[name] : MASK}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{secret.description ?? "-"}</TableCell>
                    <TableCell>{formatTime(secret.updated_at)}</TableCell>
                    <TableCell>{formatTime(secret.last_accessed_at)}</TableCell>
                    <TableCell className="text-right whitespace-nowrap">
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={t(isRevealed ? "secretsVault.actions.hide" : "secretsVault.actions.reveal", {
                          name,
                        })}
                        onClick={() => toggleReveal(name)}
                      >
                        {isRevealed ? <EyeOff /> : <Eye />}
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={t("secretsVault.actions.copy", { name })}
                        onClick={() => copy(name)}
                      >
                        <Copy />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={t("secretsVault.actions.edit", { name })}
                        onClick={() => setModal({ editing: secret })}
                      >
                        <Pencil />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={t("secretsVault.actions.delete", { name })}
                        onClick={() => setDeleting(secret)}
                      >
                        <Trash2 />
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </PageContent>

      <SecretModal
        open={modal !== null}
        editing={
          modal?.editing
            ? { name: modal.editing.secret_name, description: modal.editing.description ?? null }
            : undefined
        }
        saving={save.isPending}
        onClose={() => setModal(null)}
        onSave={handleSave}
      />

      <DeleteResourceModal
        isOpen={deleting !== null}
        title={t("secretsVault.deleteModal.title")}
        message={t("secretsVault.deleteModal.message")}
        resourceInformationTitle={t("secretsVault.deleteModal.infoTitle")}
        resourceInformation={
          deleting ? [{ label: t("secretsVault.columns.name"), value: deleting.secret_name, code: true }] : []
        }
        onCancel={() => !remove.isPending && setDeleting(null)}
        onOk={() => deleting && remove.mutate({ params: { path: { secret_name: deleting.secret_name } } })}
        confirmLoading={remove.isPending}
      />
    </>
  );
};

"use client";

import React, { useEffect } from "react";
import { useTranslation } from "react-i18next";

import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useZodForm } from "@/lib/forms/useZodForm";

import { EMPTY_SECRET, secretFormSchema, type SecretFormValues } from "./secretForm";

interface SecretModalProps {
  open: boolean;
  editing?: { name: string; description: string | null };
  saving: boolean;
  onClose: () => void;
  onSave: (values: SecretFormValues) => void;
}

export const SecretModal: React.FC<SecretModalProps> = ({ open, editing, saving, onClose, onSave }) => {
  const { t } = useTranslation();
  const form = useZodForm(secretFormSchema, { defaultValues: EMPTY_SECRET, mode: "onChange" });

  useEffect(() => {
    if (!open) return;
    form.reset(editing ? { name: editing.name, value: "", description: editing.description ?? "" } : EMPTY_SECRET);
  }, [open, editing, form]);

  return (
    <Dialog open={open} onOpenChange={(next) => !next && !saving && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>
            {editing ? t("secretsVault.modal.editTitle", { name: editing.name }) : t("secretsVault.modal.createTitle")}
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} noValidate autoComplete="off">
          <FieldGroup>
            <FormField control={form.control} name="name" label={t("secretsVault.modal.labels.name")}>
              {({ ref, ...field }) => (
                <Input
                  {...field}
                  ref={ref}
                  placeholder={t("secretsVault.modal.placeholders.name")}
                  disabled={!!editing}
                  className="font-mono"
                />
              )}
            </FormField>
            <FormField
              control={form.control}
              name="value"
              label={editing ? t("secretsVault.modal.labels.newValue") : t("secretsVault.modal.labels.value")}
            >
              {({ ref, ...field }) => (
                <Textarea
                  {...field}
                  ref={ref}
                  rows={3}
                  placeholder={t("secretsVault.modal.placeholders.value")}
                  className="font-mono"
                  spellCheck={false}
                />
              )}
            </FormField>
            <FormField control={form.control} name="description" label={t("secretsVault.modal.labels.description")}>
              {({ ref, ...field }) => (
                <Input {...field} ref={ref} placeholder={t("secretsVault.modal.placeholders.description")} />
              )}
            </FormField>
          </FieldGroup>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={saving}>
            {t("secretsVault.modal.cancel")}
          </Button>
          <Button onClick={form.handleSubmit(onSave)} disabled={saving} aria-busy={saving}>
            {editing ? t("secretsVault.modal.save") : t("secretsVault.modal.create")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

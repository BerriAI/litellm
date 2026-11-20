"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { z } from "zod";

import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import i18n from "@/i18n/i18n";

import {
  UPSTREAM_PROXY_PROTOCOLS,
  type CreateUpstreamProxy,
  type UpdateUpstreamProxy,
  type UpstreamProxy,
} from "./api";

const PROTOCOL_LABELS: Record<(typeof UPSTREAM_PROXY_PROTOCOLS)[number], string> = { http: "HTTP", https: "HTTPS" };

const PROTOCOL_ITEMS = UPSTREAM_PROXY_PROTOCOLS.map((value) => ({ value, label: PROTOCOL_LABELS[value] }));

const portInvalid = () => i18n.t("upstreamProxies.modal.portInvalid");

const proxyFormShape = {
  name: z
    .string()
    .trim()
    .regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/, { error: () => i18n.t("upstreamProxies.modal.nameInvalid") }),
  protocol: z.enum(UPSTREAM_PROXY_PROTOCOLS),
  host: z
    .string()
    .trim()
    .min(1, { error: () => i18n.t("upstreamProxies.modal.hostRequired") })
    .refine((host) => !/[\s/]/.test(host), { error: () => i18n.t("upstreamProxies.modal.hostInvalid") }),
  port: z.coerce.number().int({ error: portInvalid }).min(1, { error: portInvalid }).max(65535, { error: portInvalid }),
  username: z.string().trim(),
  password: z.string(),
  description: z.string().trim(),
};

const schema = z.object(proxyFormShape);

type FormInput = z.input<typeof schema>;
type FormValues = z.output<typeof schema>;

type Props =
  | {
      mode: "add";
      open: boolean;
      onCancel: () => void;
      onSubmit: (proxy: CreateUpstreamProxy) => Promise<void>;
    }
  | {
      mode: "edit";
      open: boolean;
      proxy: UpstreamProxy;
      onCancel: () => void;
      onSubmit: (name: string, update: UpdateUpstreamProxy) => Promise<void>;
    };

const initialValues = (props: Props): FormInput =>
  props.mode === "edit"
    ? {
        name: props.proxy.name,
        protocol: props.proxy.protocol,
        host: props.proxy.host,
        port: props.proxy.port,
        username: props.proxy.username ?? "",
        password: "",
        description: props.proxy.description ?? "",
      }
    : { name: "", protocol: "http", host: "", port: 8080, username: "", password: "", description: "" };

export const toCreateRequest = (values: FormValues): CreateUpstreamProxy => ({
  name: values.name,
  protocol: values.protocol,
  host: values.host,
  port: values.port,
  ...(values.username ? { username: values.username } : {}),
  ...(values.password ? { password: values.password } : {}),
  ...(values.description ? { description: values.description } : {}),
});

export const toUpdateRequest = (values: FormValues): UpdateUpstreamProxy => ({
  protocol: values.protocol,
  host: values.host,
  port: values.port,
  username: values.username,
  description: values.description,
  ...(values.password ? { password: values.password } : {}),
});

export default function UpstreamProxyModal(props: Props) {
  const { t } = useTranslation();
  const isEdit = props.mode === "edit";
  const form = useForm<FormInput, unknown, FormValues>({
    resolver: zodResolver(schema),
    defaultValues: initialValues(props),
  });

  const submit = form.handleSubmit(async (values) => {
    if (props.mode === "edit") {
      await props.onSubmit(props.proxy.name, toUpdateRequest(values));
      return;
    }
    await props.onSubmit(toCreateRequest(values));
  });

  return (
    <Dialog open={props.open} onOpenChange={(open) => !open && !form.formState.isSubmitting && props.onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {isEdit ? t("upstreamProxies.modal.editTitle") : t("upstreamProxies.modal.addTitle")}
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={(event) => void submit(event)}>
          <FieldGroup>
            <FormField
              control={form.control}
              name="name"
              label={t("upstreamProxies.modal.name")}
              description={isEdit ? t("upstreamProxies.modal.nameLocked") : undefined}
            >
              {({ value, ...field }) => (
                <Input
                  {...field}
                  value={String(value ?? "")}
                  placeholder={t("upstreamProxies.modal.namePlaceholder")}
                  disabled={isEdit}
                />
              )}
            </FormField>
            <FormField control={form.control} name="protocol" label={t("upstreamProxies.modal.protocol")}>
              {({ id, value, onChange }) => (
                <Select items={PROTOCOL_ITEMS} value={value} onValueChange={onChange}>
                  <SelectTrigger id={id} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PROTOCOL_ITEMS.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>
            <div className="grid grid-cols-[1fr_8rem] gap-4">
              <FormField control={form.control} name="host" label={t("upstreamProxies.modal.host")}>
                {({ value, ...field }) => (
                  <Input
                    {...field}
                    value={String(value ?? "")}
                    placeholder={t("upstreamProxies.modal.hostPlaceholder")}
                  />
                )}
              </FormField>
              <FormField control={form.control} name="port" label={t("upstreamProxies.modal.port")}>
                {({ value, ...field }) => <Input {...field} value={String(value ?? "")} inputMode="numeric" />}
              </FormField>
            </div>
            <FormField control={form.control} name="username" label={t("upstreamProxies.modal.username")}>
              {({ value, ...field }) => <Input {...field} value={String(value ?? "")} autoComplete="off" />}
            </FormField>
            <FormField
              control={form.control}
              name="password"
              label={t("upstreamProxies.modal.password")}
              description={isEdit && props.proxy.has_password ? t("upstreamProxies.modal.passwordSaved") : undefined}
            >
              {({ value, ...field }) => (
                <Input {...field} value={String(value ?? "")} type="password" autoComplete="new-password" />
              )}
            </FormField>
            <FormField control={form.control} name="description" label={t("upstreamProxies.modal.description")}>
              {({ value, ...field }) => <Textarea {...field} value={String(value ?? "")} rows={2} />}
            </FormField>
          </FieldGroup>
          <DialogFooter className="mt-6">
            <Button type="button" variant="outline" onClick={props.onCancel} disabled={form.formState.isSubmitting}>
              {t("upstreamProxies.modal.cancel")}
            </Button>
            <Button type="submit" disabled={form.formState.isSubmitting} aria-busy={form.formState.isSubmitting}>
              {isEdit ? t("upstreamProxies.modal.save") : t("upstreamProxies.modal.create")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

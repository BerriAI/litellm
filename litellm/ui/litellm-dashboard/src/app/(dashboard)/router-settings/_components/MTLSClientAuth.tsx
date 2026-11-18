import { z } from "zod";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const configSchema = z.object({
  issuer_header: z.string().default("x-amzn-mtls-clientcert-issuer"),
  subject_header: z.string().default("x-amzn-mtls-clientcert-subject"),
  allowed_clients: z.array(
    z.object({ issuer: z.string(), subject: z.string(), end_user_id: z.string().nullable().optional() }),
  ),
});
type MTLSConfig = z.infer<typeof configSchema>;
const HEADER_NAME = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;

export const mtlsClientAuthIsValid = (value: unknown) => {
  if (value == null) return true;
  const parsed = configSchema.safeParse(value);
  if (!parsed.success) return false;
  const validHeaders = HEADER_NAME.test(parsed.data.issuer_header) && HEADER_NAME.test(parsed.data.subject_header);
  const validClients = parsed.data.allowed_clients.every(
    (client) => client.issuer !== "" && client.subject !== "" && client.end_user_id !== "",
  );
  return validHeaders && validClients;
};

export function MTLSClientAuthEditor({
  value,
  disabled,
  onChange,
}: {
  value: unknown;
  disabled: boolean;
  onChange: (config: MTLSConfig | null) => void;
}) {
  const { t } = useTranslation();
  const parsed = configSchema.safeParse(value);
  if (!parsed.success) {
    return (
      <Button disabled={disabled} onClick={() => onChange(configSchema.parse({ allowed_clients: [] }))}>
        {t("routerSettings.mtlsClientAuth.enable")}
      </Button>
    );
  }
  const config = parsed.data;
  const updateClient = (index: number, patch: Partial<MTLSConfig["allowed_clients"][number]>) =>
    onChange({
      ...config,
      allowed_clients: config.allowed_clients.map((client, position) =>
        position === index ? { ...client, ...patch } : client,
      ),
    });
  return (
    <fieldset disabled={disabled} className="min-w-80 space-y-3 whitespace-normal disabled:opacity-60">
      <p className="text-xs text-muted-foreground">{t("routerSettings.mtlsClientAuth.ingressHint")}</p>
      <label className="flex flex-col gap-1 text-xs">
        {t("routerSettings.mtlsClientAuth.issuerHeader")}
        <Input
          value={config.issuer_header}
          onChange={(event) => onChange({ ...config, issuer_header: event.target.value })}
        />
      </label>
      <label className="flex flex-col gap-1 text-xs">
        {t("routerSettings.mtlsClientAuth.subjectHeader")}
        <Input
          value={config.subject_header}
          onChange={(event) => onChange({ ...config, subject_header: event.target.value })}
        />
      </label>
      {config.allowed_clients.map((client, index) => (
        <fieldset key={index} className="space-y-2 rounded-lg border p-3">
          <legend>{t("routerSettings.mtlsClientAuth.client", { number: index + 1 })}</legend>
          <label className="flex flex-col gap-1 text-xs">
            {t("routerSettings.mtlsClientAuth.issuer")}
            <Input value={client.issuer} onChange={(event) => updateClient(index, { issuer: event.target.value })} />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            {t("routerSettings.mtlsClientAuth.subject")}
            <Input value={client.subject} onChange={(event) => updateClient(index, { subject: event.target.value })} />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            {t("routerSettings.mtlsClientAuth.alias")}
            <Input
              value={client.end_user_id ?? ""}
              onChange={(event) =>
                updateClient(index, { end_user_id: event.target.value === "" ? undefined : event.target.value })
              }
            />
          </label>
          <Button
            variant="outline"
            onClick={() =>
              onChange({
                ...config,
                allowed_clients: config.allowed_clients.filter((_, position) => position !== index),
              })
            }
          >
            {t("routerSettings.mtlsClientAuth.remove", { number: index + 1 })}
          </Button>
        </fieldset>
      ))}
      {config.allowed_clients.length === 0 ? (
        <p className="text-xs text-destructive">{t("routerSettings.mtlsClientAuth.emptyWarning")}</p>
      ) : null}
      {!mtlsClientAuthIsValid(value) ? (
        <p className="text-xs text-destructive">{t("routerSettings.mtlsClientAuth.invalid")}</p>
      ) : null}
      <Button
        variant="outline"
        onClick={() =>
          onChange({ ...config, allowed_clients: [...config.allowed_clients, { issuer: "", subject: "" }] })
        }
      >
        {t("routerSettings.mtlsClientAuth.add")}
      </Button>
      <Button variant="outline" onClick={() => onChange(null)}>
        {t("routerSettings.mtlsClientAuth.disable")}
      </Button>
    </fieldset>
  );
}

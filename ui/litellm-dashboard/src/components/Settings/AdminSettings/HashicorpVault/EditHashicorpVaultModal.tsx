"use client";

import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { toast } from "@/lib/toast";
import { Modal } from "antd";
import { Eye, EyeOff } from "lucide-react";
import React, { useMemo, useState } from "react";
import { z } from "zod";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Separator } from "@/components/ui/separator";
import { useZodForm } from "@/lib/forms/useZodForm";
import { SENSITIVE_FIELDS, FIELD_LABELS } from "./constants";

interface VaultFieldGroup {
  title: string;
  subtitle?: string;
  fields: string[];
}

const FIELD_GROUPS: VaultFieldGroup[] = [
  {
    title: "Connection",
    fields: ["vault_addr", "vault_namespace", "vault_mount_name", "vault_path_prefix"],
  },
  {
    title: "Token Authentication",
    subtitle: "Use a Vault token to authenticate. Only one auth method is required.",
    fields: ["vault_token"],
  },
  {
    title: "AppRole Authentication",
    subtitle: "Use AppRole credentials to authenticate. Only one auth method is required.",
    fields: ["approle_role_id", "approle_secret_id", "approle_mount_path"],
  },
  {
    title: "TLS",
    subtitle: "Optional client certificate for mTLS.",
    fields: ["client_cert", "client_key", "vault_cert_role"],
  },
];

type VaultFormValues = Record<string, string>;

const buildSchema = (fields: readonly string[]): z.ZodType<VaultFormValues, VaultFormValues> =>
  z.object(
    Object.fromEntries(
      fields.map((name) => [
        name,
        name === "vault_addr"
          ? z.string().refine((value) => value.length === 0 || /^https?:\/\/.+/.test(value), {
              message: "Must start with http:// or https://",
            })
          : z.string(),
      ]),
    ),
  ) as unknown as z.ZodType<VaultFormValues, VaultFormValues>;

interface SecretInputProps extends React.ComponentPropsWithoutRef<"input"> {
  inputRef: React.Ref<HTMLInputElement>;
}

const SecretInput = ({ inputRef, ...props }: SecretInputProps) => {
  const [revealed, setRevealed] = useState(false);

  return (
    <InputGroup>
      <InputGroupInput ref={inputRef} type={revealed ? "text" : "password"} {...props} />
      <InputGroupAddon align="inline-end">
        <InputGroupButton
          size="icon-xs"
          aria-label={revealed ? "Hide value" : "Show value"}
          onClick={() => setRevealed((current) => !current)}
        >
          {revealed ? <EyeOff /> : <Eye />}
        </InputGroupButton>
      </InputGroupAddon>
    </InputGroup>
  );
};

interface EditHashicorpVaultModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const EditHashicorpVaultModal: React.FC<EditHashicorpVaultModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { accessToken } = useAuthorized();
  const { data } = useHashicorpVaultConfig();
  const { mutate, isPending } = useUpdateHashicorpVaultConfig(accessToken);

  const properties: Record<string, { description?: string }> = useMemo(
    () => data?.field_schema?.properties ?? {},
    [data],
  );
  const rawValues: Record<string, unknown> = useMemo(() => data?.values ?? {}, [data]);

  const visibleFields = useMemo(
    () => FIELD_GROUPS.flatMap((group) => group.fields).filter((name) => properties[name] !== undefined),
    [properties],
  );

  const seededValues = useMemo(
    () =>
      Object.fromEntries(
        visibleFields.map((name) => [name, SENSITIVE_FIELDS.has(name) ? "" : ((rawValues[name] ?? "") as string)]),
      ),
    [visibleFields, rawValues],
  );

  const schema = useMemo(() => buildSchema(visibleFields), [visibleFields]);
  const form = useZodForm(schema, { values: seededValues });

  const handleSubmit = (formValues: VaultFormValues) => {
    const config: Record<string, string> = Object.fromEntries(
      Object.entries(formValues).flatMap(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") return [[key, value]];
        if (!SENSITIVE_FIELDS.has(key)) return [[key, ""]];
        return [];
      }),
    );

    mutate(config, {
      onSuccess: () => {
        toast.success("Hashicorp Vault configuration updated successfully");
        onSuccess();
      },
      onError: (err) => {
        toast.fromError(err);
      },
    });
  };

  const handleCancel = () => {
    form.reset(seededValues);
    onCancel();
  };

  const renderField = (fieldName: string) => {
    const fieldSchema = properties[fieldName];
    if (!fieldSchema) return null;

    const isSensitive = SENSITIVE_FIELDS.has(fieldName);
    const existingValue = rawValues[fieldName];
    const hasExistingValue = isSensitive && existingValue != null && existingValue !== "";
    const placeholder = hasExistingValue ? `Leave blank to keep existing (${existingValue})` : fieldSchema?.description;

    return (
      <FormField key={fieldName} control={form.control} name={fieldName} label={FIELD_LABELS[fieldName] ?? fieldName}>
        {({ ref, ...field }) =>
          isSensitive ? (
            <SecretInput inputRef={ref} placeholder={placeholder} {...field} />
          ) : (
            <Input ref={ref} placeholder={fieldSchema?.description} {...field} />
          )
        }
      </FormField>
    );
  };

  return (
    <Modal
      title="Edit Hashicorp Vault Configuration"
      open={isVisible}
      width={700}
      footer={
        <div className="flex items-center justify-end gap-2">
          <Button type="button" variant="outline" onClick={handleCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button type="button" disabled={isPending} onClick={() => void form.handleSubmit(handleSubmit)()}>
            {isPending && <UiLoadingSpinner className="size-4 mr-1" />}
            {isPending ? "Saving..." : "Save"}
          </Button>
        </div>
      }
      onCancel={handleCancel}
    >
      <form onSubmit={form.handleSubmit(handleSubmit)}>
        {FIELD_GROUPS.map((group, index) => (
          <div key={group.title}>
            {index > 0 && <Separator className="my-6" />}
            <h5 className="mb-1 text-base font-semibold text-foreground">{group.title}</h5>
            {group.subtitle && <p className="mb-4 text-sm text-muted-foreground">{group.subtitle}</p>}
            <FieldGroup>{group.fields.map(renderField)}</FieldGroup>
          </div>
        ))}
      </form>
    </Modal>
  );
};

export default EditHashicorpVaultModal;

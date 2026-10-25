import React from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useFormContext } from "react-hook-form";
import { FormField } from "@/components/shared/form/FormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { CoordinationField } from "./coordinationRedisFields";
import type { CoordinationFormValues } from "./coordinationRedisUtils";

export const SECRET_ALREADY_SET_PLACEHOLDER = "Already set. Enter a new value to replace it.";

interface CoordinationRedisFormFieldProps {
  field: CoordinationField;
  isSecretConfigured: boolean;
}

const CoordinationRedisFormField: React.FC<CoordinationRedisFormFieldProps> = ({ field, isSecretConfigured }) => {
  const { t } = useLanguage();
  const form = useFormContext<CoordinationFormValues>();
  const placeholder = t(isSecretConfigured ? SECRET_ALREADY_SET_PLACEHOLDER : field.helpText);

  return (
    <FormField
      control={form.control}
      name={field.name}
      label={t(field.label)}
      description={t(field.helpText)}
      // 仅翻译本地规则产生的错误，服务端同名文案也保持原样。
      renderError={(error) => (error.type === "local-validation" && error.message ? t(error.message) : error.message)}
    >
      {({ ref, value, onChange, ...rest }) => {
        if (field.type === "boolean") {
          return <Switch {...rest} checked={value === true} onCheckedChange={(checked) => onChange(checked)} />;
        }
        if (field.type === "password") {
          return (
            <PasswordInput
              {...rest}
              ref={ref}
              value={typeof value === "string" ? value : ""}
              onChange={onChange}
              placeholder={placeholder}
              autoComplete="new-password"
            />
          );
        }
        if (field.type === "list") {
          return (
            <Textarea
              {...rest}
              ref={ref}
              rows={4}
              value={typeof value === "string" ? value : ""}
              onChange={onChange}
              placeholder={placeholder}
            />
          );
        }
        return (
          <Input
            {...rest}
            ref={ref}
            inputMode={field.type === "integer" ? "numeric" : undefined}
            value={typeof value === "string" ? value : ""}
            onChange={onChange}
            placeholder={placeholder}
          />
        );
      }}
    </FormField>
  );
};

export default CoordinationRedisFormField;

import React from "react";
import { useForm } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { CircleCheck, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

interface AlertingSetting {
  field_name: string;
  field_description: string;
  field_type: string;
  field_value: any;
  stored_in_db: boolean | null;
  premium_field: boolean;
}

interface DynamicFormProps {
  alertingSettings: AlertingSetting[];
  handleInputChange: (fieldName: string, newValue: any) => void;
  handleResetField: (fieldName: string, index: number) => void;
  handleSubmit: (formValues: Record<string, any>) => void;
  premiumUser: boolean;
}

type AlertingFormValues = Record<string, string | boolean>;

const DynamicForm: React.FC<DynamicFormProps> = ({
  alertingSettings,
  handleInputChange,
  handleResetField,
  handleSubmit,
  premiumUser,
}) => {
  const { t } = useTranslation();
  const form = useForm<AlertingFormValues>({ defaultValues: {} });

  const onFinish = (formData: AlertingFormValues) => {
    const isEmpty = Object.entries(formData).every(([, value]) => {
      if (typeof value === "boolean") {
        return false;
      }
      return value === "" || value === null || value === undefined;
    });
    if (!isEmpty) {
      handleSubmit(formData);
    }
  };

  const handleNumericChange = (setting: AlertingSetting, raw: string) => {
    form.setValue(setting.field_name, raw);
    handleInputChange(setting.field_name, raw === "" ? null : Number(raw));
  };

  const handleTextChange = (setting: AlertingSetting, event: React.ChangeEvent<HTMLInputElement>) => {
    form.setValue(setting.field_name, event.target.value);
    handleInputChange(setting.field_name, event);
  };

  const handleToggle = (setting: AlertingSetting, checked: boolean) => {
    form.setValue(setting.field_name, checked);
    handleInputChange(setting.field_name, checked);
  };

  const renderControl = (setting: AlertingSetting) => {
    if (setting.field_type === "Integer" || setting.field_type === "Float") {
      return (
        <Input
          type="number"
          step={setting.field_type === "Integer" ? 1 : "any"}
          value={setting.field_value ?? ""}
          onChange={(event) => handleNumericChange(setting, event.target.value)}
        />
      );
    }
    if (setting.field_type === "Boolean") {
      return (
        <Switch
          aria-label={setting.field_name}
          checked={setting.field_value}
          onCheckedChange={(checked) => handleToggle(setting, checked)}
        />
      );
    }
    return <Input value={setting.field_value ?? ""} onChange={(event) => handleTextChange(setting, event)} />;
  };

  return (
    <form className="max-w-5xl space-y-4" onSubmit={form.handleSubmit(onFinish)} noValidate>
      {alertingSettings.map((value, index) => (
        <div key={index} className="flex flex-wrap items-center gap-3 border-b border-border pb-4">
          <div className="min-w-0 basis-full sm:basis-64 sm:flex-1">
            <p className="break-words text-sm font-medium">{value.field_name}</p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{value.field_description}</p>
          </div>
          {value.premium_field && !premiumUser ? (
            <div className="shrink-0">
              <Button className="flex items-center justify-center">
                <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                  {t("loggingAndAlerts.alerting.enterpriseFeature")}
                </a>
              </Button>
            </div>
          ) : (
            <div className="min-w-0 basis-40 flex-1 sm:max-w-64">{renderControl(value)}</div>
          )}
          <div className="shrink-0">
            {value.stored_in_db == true ? (
              <Badge variant="secondary">
                <CircleCheck />
                {t("loggingAndAlerts.alerting.storage.inDb")}
              </Badge>
            ) : value.stored_in_db == false ? (
              <Badge variant="outline">{t("loggingAndAlerts.alerting.storage.inConfig")}</Badge>
            ) : (
              <Badge variant="outline">{t("loggingAndAlerts.alerting.storage.notSet")}</Badge>
            )}
          </div>
          <div className="shrink-0">
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label={t("loggingAndAlerts.alerting.resetField", { field: value.field_name })}
              onClick={() => handleResetField(value.field_name, index)}
              className="text-destructive"
            >
              <Trash2 className="size-5" />
            </Button>
          </div>
        </div>
      ))}
      <div>
        <Button type="submit">{t("loggingAndAlerts.alerting.submit")}</Button>
      </div>
    </form>
  );
};

export default DynamicForm;

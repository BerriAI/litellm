import React, { useEffect, useState } from "react";
import { useController } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { getCallbacksCall, getRouterSettingsCall, setCallbacksCall } from "../networking";
import RouterSettingsForm, { RouterSettingsFormValue } from "./RouterSettingsForm";
import { RouterSettingsSource, SOURCE_UNKNOWN_HINT } from "./ConfigOwned";
import { toRouterSettingsPayload } from "./mapper";
import { routerSettingsSchema } from "./schema";

interface RouterSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const RouterSettings: React.FC<RouterSettingsProps> = ({ accessToken, userRole, userID }) => {
  const { t } = useTranslation();
  const { control, getValues, setValue } = useZodForm(routerSettingsSchema, {
    defaultValues: { selectedStrategy: null, enableTagFiltering: false },
  });
  const { field: strategy } = useController({ control, name: "selectedStrategy" });
  const { field: tagFiltering } = useController({ control, name: "enableTagFiltering" });
  const [routerSettings, setRouterSettings] = useState<RouterSettingsFormValue["routerSettings"]>({});
  const [availableRoutingStrategies, setAvailableRoutingStrategies] = useState<string[]>([]);
  const [routerFieldsMetadata, setRouterFieldsMetadata] = useState<{ [key: string]: any }>({});
  const [routingStrategyDescriptions, setRoutingStrategyDescriptions] = useState<{ [key: string]: string }>({});
  const [routerSettingsSource, setRouterSettingsSource] = useState<RouterSettingsSource | null>(null);

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      let router_settings = data.router_settings;
      if ("model_group_retry_policy" in router_settings) {
        delete router_settings["model_group_retry_policy"];
      }
      // Set initial selected strategy
      const initialStrategy = router_settings.routing_strategy || null;
      setRouterSettings(router_settings);
      setValue("selectedStrategy", initialStrategy);
    });
    getRouterSettingsCall(accessToken)
      .catch(() => {
        setRouterSettingsSource(null);
        return null;
      })
      .then((data) => {
        if (!data) {
          return;
        }
        setRouterSettingsSource(data.source ?? {});
        if (data.fields) {
          // Build metadata map for easy lookup
          const fieldsMap: { [key: string]: any } = {};
          data.fields.forEach((field: any) => {
            fieldsMap[field.field_name] = {
              ui_field_name: field.ui_field_name,
              field_description: field.field_description,
              options: field.options,
              link: field.link,
            };
          });
          setRouterFieldsMetadata(fieldsMap);

          // Extract routing strategies from the routing_strategy field's options
          const routingStrategyField = data.fields.find((field: any) => field.field_name === "routing_strategy");
          if (routingStrategyField?.options) {
            setAvailableRoutingStrategies(routingStrategyField.options);
          }

          // Store routing strategy descriptions
          if (data.routing_strategy_descriptions) {
            setRoutingStrategyDescriptions(data.routing_strategy_descriptions);
          }

          // Set enable_tag_filtering value
          const tagFilteringField = data.fields.find((field: any) => field.field_name === "enable_tag_filtering");
          if (tagFilteringField?.field_value !== null && tagFilteringField?.field_value !== undefined) {
            setValue("enableTagFiltering", tagFilteringField.field_value);
          }
        }
      });
  }, [accessToken, userRole, userID, setValue]);

  const handleSaveChanges = async () => {
    if (!accessToken || routerSettingsSource === null) {
      return;
    }

    const values = routerSettingsSchema.parse(getValues());
    const payload = toRouterSettingsPayload(
      { ...values, routerSettings },
      routerSettingsSource,
      (name) => document.querySelector<HTMLInputElement>(`input[name="${name}"]`)?.value,
    );

    try {
      await setCallbacksCall(accessToken, payload);
      toast.success(t("routerSettingsForm.page.updated"));
    } catch (error) {
      toast.fromError(t("routerSettingsForm.page.updateFailed") + error);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full">
      <RouterSettingsForm
        value={{ routerSettings, selectedStrategy: strategy.value, enableTagFiltering: tagFiltering.value }}
        onChange={({ selectedStrategy, enableTagFiltering }) => {
          setValue(strategy.name, selectedStrategy);
          setValue(tagFiltering.name, enableTagFiltering);
        }}
        routerFieldsMetadata={routerFieldsMetadata}
        availableRoutingStrategies={availableRoutingStrategies}
        routingStrategyDescriptions={routingStrategyDescriptions}
        routerSettingsSource={routerSettingsSource ?? {}}
      />

      {/* Actions - Sticky at bottom */}
      <div className="border-t border-border pt-6 flex justify-end gap-3">
        <Button variant="outline" onClick={() => window.location.reload()}>
          {t("routerSettingsForm.page.reset")}
        </Button>
        <Button
          onClick={handleSaveChanges}
          disabled={routerSettingsSource === null}
          title={routerSettingsSource === null ? t(SOURCE_UNKNOWN_HINT) : undefined}
        >
          {t("routerSettingsForm.page.save")}
        </Button>
      </div>
    </div>
  );
};

export default RouterSettings;
export { RouterSettingsForm };
export type { RouterSettingsFormValue };

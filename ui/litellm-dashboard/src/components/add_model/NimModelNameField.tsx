import { useEffect, useMemo, useState, type ComponentProps } from "react";
import { useFormContext, useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { apiClient } from "../networking";
import { resolveLitellmProviderSlug } from "../provider_info_helpers";
import type { MountedFormValues } from "../common_components/MountedFormField";
import { Button } from "@/components/ui/button";
import LiteLLMModelNameField from "./litellm_model_name";

type DiscoveryResponse = {
  status: "success" | "unavailable";
  models: string[];
};

type Props = ComponentProps<typeof LiteLLMModelNameField> & {
  accessToken: string | null;
  canDiscover: boolean;
};

export default function NimModelNameField({ accessToken, canDiscover, ...props }: Props) {
  const { t } = useTranslation();
  const { control } = useFormContext<MountedFormValues>();
  const credential = useWatch({ control, name: "litellm_credential_name" });
  const apiKey = useWatch({ control, name: "api_key" });
  const apiBase = useWatch({ control, name: "api_base" });
  const enabled =
    canDiscover &&
    props.selectedProvider !== null &&
    resolveLitellmProviderSlug(props.selectedProvider) === "nvidia_nim";
  const connection = useMemo(
    () => ({
      credential_name: typeof credential === "string" && credential ? credential : undefined,
      credential_values: credential
        ? undefined
        : {
            api_key: typeof apiKey === "string" && apiKey ? apiKey : undefined,
            api_base: typeof apiBase === "string" && apiBase ? apiBase : undefined,
          },
    }),
    [credential, apiKey, apiBase],
  );
  const [refresh, setRefresh] = useState<{ connection: typeof connection; attempt: number }>();
  const attempt = refresh?.connection === connection ? refresh.attempt : 0;
  const shouldDiscover = Boolean(connection.credential_name) || refresh?.connection === connection;
  const [result, setResult] = useState<{
    connection: typeof connection;
    accessToken: string | null;
    attempt: number;
    response: DiscoveryResponse;
  }>();
  const current =
    result?.connection === connection && result.accessToken === accessToken && result.attempt === attempt
      ? result.response
      : undefined;

  useEffect(() => {
    if (!enabled || !shouldDiscover) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await apiClient.post<DiscoveryResponse>("/credentials/nvidia_nim/models", {
          accessToken,
          body: connection,
          signal: controller.signal,
        });
        if (!controller.signal.aborted) {
          const nextResult = { connection, accessToken, attempt, response };
          setResult(nextResult);
        }
      } catch {
        if (!controller.signal.aborted) {
          const response: DiscoveryResponse = { status: "unavailable", models: [] };
          const nextResult = { connection, accessToken, attempt, response };
          setResult(nextResult);
        }
      }
    }, 500);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [enabled, shouldDiscover, connection, accessToken, attempt]);

  return (
    <>
      <LiteLLMModelNameField
        {...props}
        providerModels={enabled && current?.status === "success" ? current.models : props.providerModels}
      />
      {enabled && (
        <div className="mb-4 text-sm text-muted-foreground">
          <p role="status">
            {!shouldDiscover && t("addModel.nimModelDiscovery.ready")}
            {shouldDiscover && !current && t("addModel.nimModelDiscovery.loading")}
            {current?.status === "success" && t("addModel.nimModelDiscovery.success")}
            {current?.status === "unavailable" && t("addModel.nimModelDiscovery.unavailable")}
          </p>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={shouldDiscover && !current}
            onClick={() => setRefresh((value) => ({ connection, attempt: (value?.attempt ?? 0) + 1 }))}
          >
            {t("addModel.nimModelDiscovery.refresh")}
          </Button>
        </div>
      )}
    </>
  );
}

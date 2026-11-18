import { useEffect, useMemo, useState, type ComponentProps } from "react";
import { useFormContext, useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { apiClient } from "../networking";
import { Providers, resolveLitellmProviderSlug } from "../provider_info_helpers";
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

function discoveryProvider(selectedProvider: string | null) {
  if (selectedProvider === null) return undefined;
  if (selectedProvider === Providers.OLLAMA_CHAT) return "ollama_chat";
  const slug = resolveLitellmProviderSlug(selectedProvider);
  switch (slug) {
    case "nvidia_nim":
    case "ollama":
    case "ollama_chat":
      return slug;
    default:
      return undefined;
  }
}

function optionalText(value: unknown) {
  return typeof value === "string" && value ? value : undefined;
}

export default function UpstreamModelNameField({ accessToken, canDiscover, ...props }: Props) {
  const { t } = useTranslation();
  const { control } = useFormContext<MountedFormValues>();
  const credential = useWatch({ control, name: "litellm_credential_name" });
  const apiKey = useWatch({ control, name: "api_key" });
  const apiBase = useWatch({ control, name: "api_base" });
  const provider = discoveryProvider(props.selectedProvider);
  const enabled = canDiscover && provider !== undefined;
  const translationKey = provider === "nvidia_nim" ? "addModel.nimModelDiscovery" : "addModel.ollamaModelDiscovery";
  const connection = useMemo(
    () => ({
      provider,
      body: {
        credential_name: optionalText(credential),
        credential_values: credential
          ? undefined
          : {
              api_key: optionalText(apiKey),
              api_base: optionalText(apiBase),
            },
      },
    }),
    [provider, credential, apiKey, apiBase],
  );
  const [refresh, setRefresh] = useState<{ connection: typeof connection; attempt: number }>();
  const attempt = refresh?.connection === connection ? refresh.attempt : 0;
  const shouldDiscover = Boolean(connection.body.credential_name) || refresh?.connection === connection;
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
  const status = !shouldDiscover ? "ready" : current?.status ?? "loading";
  const upstreamModels = enabled && current?.status === "success" ? current.models : undefined;

  useEffect(() => {
    if (!enabled || !shouldDiscover) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await apiClient.post<DiscoveryResponse>(`/credentials/${connection.provider}/models`, {
          accessToken,
          body: connection.body,
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
        providerModels={upstreamModels ?? props.providerModels}
        hasUpstreamModels={upstreamModels !== undefined}
      />
      {enabled && (
        <div className="mb-4 text-sm text-muted-foreground">
          <p role="status">{t(`${translationKey}.${status}`)}</p>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={shouldDiscover && !current}
            onClick={() => setRefresh((value) => ({ connection, attempt: (value?.attempt ?? 0) + 1 }))}
          >
            {t(`${translationKey}.refresh`)}
          </Button>
        </div>
      )}
    </>
  );
}

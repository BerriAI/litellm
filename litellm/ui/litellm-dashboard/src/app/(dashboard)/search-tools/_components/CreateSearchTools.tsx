import { isAdminRole } from "@/utils/roles";
import { useQuery } from "@tanstack/react-query";
import { CircleHelp } from "lucide-react";
import React, { useCallback, useMemo, useState } from "react";
import { useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { z } from "zod";
import { Logo } from "@/components/molecules/logo/Logo";
import { toast } from "@/lib/toast";
import { createSearchTool, fetchAvailableSearchProviders } from "@/components/networking";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import SearchConnectionTest from "./SearchConnectionTest";
import { buildSearchToolPayload } from "./searchToolPayload";
import { AvailableSearchProvider, SearchTool } from "./types";
import bingLogo from "../../../../../public/assets/logos/bing.png";
import dataforseoLogo from "../../../../../public/assets/logos/dataforseo.png";
import exaAiLogo from "../../../../../public/assets/logos/exa_ai.png";
import googlePseLogo from "../../../../../public/assets/logos/google_pse.png";
import nimbleLogo from "../../../../../public/assets/logos/nimble.png";
import parallelAiLogo from "../../../../../public/assets/logos/parallel_ai.png";
import perplexityLogo from "../../../../../public/assets/logos/perplexity.png";
import tavilyLogo from "../../../../../public/assets/logos/tavily.png";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const searchProviderLogoMap: Record<string, string> = {
  perplexity: perplexityLogo.src,
  tavily: tavilyLogo.src,
  parallel_ai: parallelAiLogo.src,
  exa_ai: exaAiLogo.src,
  google_pse: googlePseLogo.src,
  dataforseo: dataforseoLogo.src,
  nimble: nimbleLogo.src,
  bing_grounding: bingLogo.src,
};

interface SearchProviderLabelProps {
  providerName: string;
  displayName: string;
}

export const SearchProviderLabel: React.FC<SearchProviderLabelProps> = ({ providerName, displayName }) => (
  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
    <Logo src={searchProviderLogoMap[providerName]} label={displayName} className="w-5 h-5 object-contain" />
    <span>{displayName}</span>
  </div>
);

const createSearchToolShape = {
  search_tool_name: z
    .string()
    .min(1, { error: () => i18n.t("searchTools.createForm.errors.nameRequired") })
    .regex(/^[a-zA-Z0-9_-]+$/, { error: () => i18n.t("searchTools.createForm.errors.nameInvalid") }),
  search_provider: z
    .string()
    .nullable()
    .pipe(
      z
        .string({ error: () => i18n.t("searchTools.createForm.errors.providerRequired") })
        .min(1, { error: () => i18n.t("searchTools.createForm.errors.providerRequired") }),
    ),
  api_key: z.string().optional(),
  description: z.string().optional(),
  input_cost_per_query: z
    .number()
    .nonnegative({ error: () => i18n.t("searchTools.createForm.errors.costNegative") })
    .optional(),
};

const createSearchToolSchema = z.object(createSearchToolShape);

type CreateSearchToolFormValues = z.infer<typeof createSearchToolSchema>;

const EMPTY_VALUES: z.input<typeof createSearchToolSchema> = { search_tool_name: "", search_provider: null };

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

interface CreateSearchToolProps {
  userRole: string;
  accessToken: string | null;
  onCreateSuccess: (newSearchTool: SearchTool) => void;
  isModalVisible: boolean;
  setModalVisible: (visible: boolean) => void;
}

const CreateSearchTool: React.FC<CreateSearchToolProps> = ({
  userRole,
  accessToken,
  onCreateSuccess,
  isModalVisible,
  setModalVisible,
}) => {
  const { t } = useTranslation();
  const form = useZodForm(createSearchToolSchema, { defaultValues: EMPTY_VALUES });
  const [isLoading, setIsLoading] = useState(false);
  const [isTestModalVisible, setIsTestModalVisible] = useState(false);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");
  const [watchedProvider, watchedApiKey] = useWatch({
    control: form.control,
    name: ["search_provider", "api_key"],
  });

  const { data: providersResponse, isLoading: isLoadingProviders } = useQuery({
    queryKey: ["searchProviders"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchAvailableSearchProviders(accessToken);
    },
    enabled: !!accessToken && isModalVisible,
  }) as { data: { providers: AvailableSearchProvider[] }; isLoading: boolean };

  const availableProviders = providersResponse?.providers;
  const providerNames = useMemo(
    () => (availableProviders ?? []).map((provider) => provider.provider_name),
    [availableProviders],
  );
  const providerLabel = useCallback(
    (providerName: string) =>
      (availableProviders ?? []).find((provider) => provider.provider_name === providerName)?.ui_friendly_name ??
      providerName,
    [availableProviders],
  );

  const handleCreate = async (formValues: CreateSearchToolFormValues) => {
    setIsLoading(true);
    try {
      const payload = buildSearchToolPayload(formValues);

      if (accessToken != null) {
        const response = await createSearchTool(accessToken, payload);

        toast.success(t("searchTools.createForm.toasts.created"));
        form.reset(EMPTY_VALUES);
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      toast.error(t("searchTools.createForm.toasts.createFailed", { error: String(error) }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    form.reset(EMPTY_VALUES);
    setModalVisible(false);
  };

  const handleTestConnection = async () => {
    const isValid = await form.trigger(["search_provider", "api_key"]);
    if (!isValid) {
      toast.error(t("searchTools.createForm.toasts.testMissingFields"));
      return;
    }

    setIsTestingConnection(true);
    setConnectionTestId(`test-${Date.now()}`);
    setIsTestModalVisible(true);
  };

  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <div className="flex items-center space-x-3 pb-4 border-b border-border">
            <span className="text-2xl">🔍</span>
            <DialogTitle className="text-xl font-semibold text-foreground">
              {t("searchTools.createForm.title")}
            </DialogTitle>
          </div>
        </DialogHeader>
        <div className="mt-6">
          <TooltipProvider>
            <form onSubmit={form.handleSubmit(handleCreate)} className="space-y-6">
              <FieldGroup>
                <FormField
                  control={form.control}
                  name="search_tool_name"
                  label={labelWithHint(t("searchTools.createForm.labels.name"), t("searchTools.createForm.hints.name"))}
                >
                  {({ ref, ...field }) => (
                    <Input
                      {...field}
                      ref={ref}
                      placeholder={t("searchTools.createForm.placeholders.name")}
                      className="rounded-lg"
                    />
                  )}
                </FormField>

                <FormField
                  control={form.control}
                  name="search_provider"
                  label={labelWithHint(
                    t("searchTools.createForm.labels.provider"),
                    t("searchTools.createForm.hints.provider"),
                  )}
                >
                  {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                    <Combobox
                      items={providerNames}
                      itemToStringLabel={providerLabel}
                      value={value}
                      onValueChange={onChange}
                    >
                      <ComboboxInput
                        id={id}
                        aria-invalid={ariaInvalid}
                        aria-describedby={ariaDescribedBy}
                        placeholder={t("searchTools.createForm.placeholders.provider")}
                        className="h-10 w-full rounded-lg"
                        disabled={isLoadingProviders}
                        showClear={value != null && value !== ""}
                      />
                      <ComboboxContent>
                        <ComboboxEmpty>{t("searchTools.createForm.noProviders")}</ComboboxEmpty>
                        <ComboboxList>
                          {(providerName: string) => (
                            <ComboboxItem key={providerName} value={providerName}>
                              <SearchProviderLabel
                                providerName={providerName}
                                displayName={providerLabel(providerName)}
                              />
                            </ComboboxItem>
                          )}
                        </ComboboxList>
                      </ComboboxContent>
                    </Combobox>
                  )}
                </FormField>

                <FormField
                  control={form.control}
                  name="api_key"
                  label={labelWithHint(
                    t("searchTools.createForm.labels.apiKey"),
                    t("searchTools.createForm.hints.apiKey"),
                  )}
                >
                  {({ ref, value, ...field }) => (
                    <PasswordInput
                      {...field}
                      ref={ref}
                      value={value ?? ""}
                      placeholder={t("searchTools.createForm.placeholders.apiKey")}
                      groupClassName="h-10 rounded-lg"
                    />
                  )}
                </FormField>

                <FormField
                  control={form.control}
                  name="input_cost_per_query"
                  label={labelWithHint(t("searchTools.createForm.labels.cost"), t("searchTools.createForm.hints.cost"))}
                >
                  {({ ref, value, onChange, ...field }) => (
                    <Input
                      {...field}
                      ref={ref}
                      type="number"
                      step="any"
                      value={value ?? ""}
                      onChange={(event) => onChange(event.target.value === "" ? undefined : event.target.valueAsNumber)}
                      placeholder={t("searchTools.createForm.placeholders.cost")}
                      className="rounded-lg"
                    />
                  )}
                </FormField>

                <FormField
                  control={form.control}
                  name="description"
                  label={t("searchTools.createForm.labels.description")}
                >
                  {({ ref, value, ...field }) => (
                    <Textarea
                      {...field}
                      ref={ref}
                      value={value ?? ""}
                      rows={3}
                      placeholder={t("searchTools.createForm.placeholders.description")}
                      className="rounded-lg"
                    />
                  )}
                </FormField>
              </FieldGroup>

              <div className="flex justify-between items-center pt-6 border-t border-border">
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <a
                        className="text-sm text-info hover:underline"
                        href="https://github.com/BerriAI/litellm/issues"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {t("searchTools.createForm.help")}
                      </a>
                    }
                  />
                  <TooltipContent>{t("searchTools.createForm.helpTooltip")}</TooltipContent>
                </Tooltip>
                <div className="flex gap-2">
                  <Button type="submit" variant="outline" onClick={handleTestConnection} disabled={isTestingConnection}>
                    {isTestingConnection && <UiLoadingSpinner className="size-4" />}
                    {t("searchTools.createForm.testConnection")}
                  </Button>
                  <Button type="submit" variant="outline" disabled={isLoading}>
                    {isLoading && <UiLoadingSpinner className="size-4" />}
                    {t("searchTools.createForm.submit")}
                  </Button>
                </div>
              </div>
            </form>
          </TooltipProvider>
        </div>

        <Dialog
          open={isTestModalVisible}
          onOpenChange={(open) => {
            if (!open) {
              setIsTestModalVisible(false);
              setIsTestingConnection(false);
            }
          }}
        >
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
            <DialogHeader>
              <DialogTitle>{t("searchTools.createForm.testResultsTitle")}</DialogTitle>
            </DialogHeader>
            {isTestModalVisible && accessToken && (
              <SearchConnectionTest
                key={connectionTestId}
                litellmParams={{
                  search_provider: watchedProvider ?? undefined,
                  api_key: watchedApiKey,
                  api_base: undefined,
                }}
                accessToken={accessToken}
                onTestComplete={() => setIsTestingConnection(false)}
              />
            )}
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setIsTestModalVisible(false);
                  setIsTestingConnection(false);
                }}
              >
                {t("searchTools.createForm.close")}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
};

export default CreateSearchTool;

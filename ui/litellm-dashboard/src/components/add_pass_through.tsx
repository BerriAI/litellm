"use client";

import React, { useState } from "react";
import { CircleHelp, Info, Plug } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { z } from "zod";

import { createPassThroughEndpoint } from "./networking";
import NumericalInput from "./shared/numerical_input";
import KeyValueInput, { type KeyValuePair } from "./key_value_input";
import QueryParamInput from "./query_param_input";
import { passThroughItem } from "./PassThroughSettings/PassThroughSettings";
import RoutePreview from "./route_preview";
import { toast } from "@/lib/toast";
import PassThroughSecuritySection from "./common_components/PassThroughSecuritySection";
import PassThroughGuardrailsSection from "./common_components/PassThroughGuardrailsSection";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Card } from "@/components/ui/card";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import i18n from "@/i18n/i18n";
import { codeSamples } from "@/i18n/untranslated/codeSamples";

const HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"] as const;
const HTTP_METHOD_OPTIONS = HTTP_METHODS.map((method) => ({ label: method, value: method }));

type GuardrailSettings = Record<string, { request_fields?: string[]; response_fields?: string[] } | null>;

const keyValuePairsSchema = z.array(z.tuple([z.string(), z.string()]));

const passThroughFormSchema = z.object({
  path: z
    .string()
    .min(1, { error: () => i18n.t("addPassThrough.validation.pathRequired") })
    .regex(/^\//, { error: () => i18n.t("addPassThrough.validation.pathRequired") }),
  target: z
    .string()
    .min(1, { error: () => i18n.t("addPassThrough.validation.targetRequired") })
    .pipe(z.url({ error: () => i18n.t("addPassThrough.validation.invalidUrl") })),
  methods: z.array(z.string()).optional(),
  include_subpath: z.boolean(),
  headers: keyValuePairsSchema,
  default_query_params: keyValuePairsSchema.optional(),
  auth: z.boolean(),
  timeout: z.string().optional(),
  cost_per_request: z.string().optional(),
});

type PassThroughFormValues = z.output<typeof passThroughFormSchema>;

const emptyFormValues = {
  path: "",
  target: "",
  methods: undefined,
  include_subpath: true,
  headers: [],
  default_query_params: undefined,
  auth: true,
  timeout: undefined,
  cost_per_request: undefined,
} as unknown as PassThroughFormValues;

const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const optionalText = (raw: string): string | undefined => (raw === "" ? undefined : raw);

const toRecord = (pairs: readonly KeyValuePair[]): Record<string, string> =>
  Object.fromEntries(pairs.filter(([name]) => name !== ""));

const optionalRecord = (pairs: readonly KeyValuePair[] | undefined): Record<string, string> | undefined => {
  const record = toRecord(pairs ?? []);
  return Object.keys(record).length > 0 ? record : undefined;
};

interface AddFallbacksProps {
  accessToken: string;
  passThroughItems: passThroughItem[];
  setPassThroughItems: React.Dispatch<React.SetStateAction<passThroughItem[]>>;
}

const AddPassThroughEndpoint: React.FC<AddFallbacksProps> = ({
  accessToken,
  setPassThroughItems,
  passThroughItems,
}) => {
  const { t } = useTranslation();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [guardrails, setGuardrails] = useState<GuardrailSettings>({});
  const form = useZodForm(passThroughFormSchema, { defaultValues: emptyFormValues });

  const pathValue = useWatch({ control: form.control, name: "path" });
  const targetValue = useWatch({ control: form.control, name: "target" });
  const includeSubpath = useWatch({ control: form.control, name: "include_subpath" });
  const selectedMethods = useWatch({ control: form.control, name: "methods" }) ?? [];

  const handleCancel = () => {
    form.reset(emptyFormValues);
    setGuardrails({});
    setIsModalVisible(false);
  };

  const addPassThrough = async (values: PassThroughFormValues) => {
    setIsLoading(true);
    try {
      const formValues = {
        path: values.path,
        target: values.target,
        methods: values.methods,
        include_subpath: values.include_subpath,
        headers: toRecord(values.headers),
        default_query_params: optionalRecord(values.default_query_params),
        auth: values.auth,
        timeout: values.timeout,
        cost_per_request: values.cost_per_request,
        ...(Object.keys(guardrails).length > 0 ? { guardrails } : {}),
      };

      const response = await createPassThroughEndpoint(accessToken, formValues);
      const createdEndpoint = response.endpoints[0];

      setPassThroughItems([...passThroughItems, createdEndpoint]);

      toast.success(t("addPassThrough.toasts.created"));
      form.reset(emptyFormValues);
      setGuardrails({});
      setIsModalVisible(false);
    } catch (error) {
      toast.fromError(t("addPassThrough.toasts.createFailed", { error: String(error) }));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <TooltipProvider>
      <div>
        <Button className="mx-auto mb-4 mt-4" onClick={() => setIsModalVisible(true)}>
          {t("addPassThrough.openButton")}
        </Button>
        <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
          <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[1000px]">
            <DialogHeader>
              <div className="flex items-center space-x-3 border-b border-border pb-4">
                <Plug className="size-5 text-info" />
                <DialogTitle className="text-xl font-semibold text-foreground">{t("addPassThrough.title")}</DialogTitle>
              </div>
            </DialogHeader>
            <div className="mt-6">
              <Alert variant="info" className="mb-6">
                <Info />
                <AlertTitle>{t("addPassThrough.intro.title")}</AlertTitle>
                <AlertDescription>{t("addPassThrough.intro.description")}</AlertDescription>
              </Alert>

              <form onSubmit={form.handleSubmit(addPassThrough)} className="space-y-6">
                <Card className="block p-5">
                  <h3 className="mb-2 text-lg font-semibold text-foreground">{t("addPassThrough.route.title")}</h3>
                  <p className="mb-5 text-sm text-muted-foreground">{t("addPassThrough.route.description")}</p>

                  <div className="space-y-5">
                    <FormField
                      control={form.control}
                      name="path"
                      label={t("addPassThrough.route.pathPrefix")}
                      description={t("addPassThrough.route.pathPrefixExample")}
                    >
                      {({ value, onChange, ...field }) => (
                        <Input
                          {...field}
                          placeholder={codeSamples.passThrough.targetPrefix}
                          value={value ?? ""}
                          onChange={(event) => {
                            const raw = event.target.value;
                            onChange(raw && !raw.startsWith("/") ? "/" + raw : raw);
                          }}
                        />
                      )}
                    </FormField>

                    <FormField
                      control={form.control}
                      name="target"
                      label={t("addPassThrough.route.targetUrl")}
                      description={t("addPassThrough.route.targetUrlExample")}
                    >
                      {({ value, ...field }) => (
                        <Input {...field} placeholder="https://engine.prod.bria-api.com" value={value ?? ""} />
                      )}
                    </FormField>

                    <FormField
                      control={form.control}
                      name="methods"
                      label={labelWithHint(
                        t("addPassThrough.route.httpMethods"),
                        t("addPassThrough.route.httpMethodsHint"),
                      )}
                      description={
                        selectedMethods.length === 0
                          ? t("addPassThrough.route.allMethodsDefault")
                          : t("addPassThrough.route.onlyMethods", { methods: selectedMethods.join(", ") })
                      }
                    >
                      {({ value, onChange, ref: _ref, ...field }) => (
                        <Select multiple items={HTTP_METHOD_OPTIONS} value={value ?? []} onValueChange={onChange}>
                          <SelectTrigger {...field} className="w-full">
                            <SelectValue placeholder={t("addPassThrough.route.methodsPlaceholder")}>
                              {(selected: string[]) =>
                                selected.length === 0
                                  ? t("addPassThrough.route.methodsPlaceholder")
                                  : selected.join(", ")
                              }
                            </SelectValue>
                          </SelectTrigger>
                          <SelectContent>
                            {HTTP_METHODS.map((method) => (
                              <SelectItem key={method} value={method} title={method}>
                                {method}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      )}
                    </FormField>

                    <div className="flex items-center justify-between py-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">
                          {t("addPassThrough.route.includeSubpaths")}
                        </div>
                        <div className="mt-0.5 text-xs text-muted-foreground">
                          {t("addPassThrough.route.includeSubpathsDescription")}
                        </div>
                      </div>
                      <FormField control={form.control} name="include_subpath">
                        {({ value, onChange, ref: _ref, ...field }) => (
                          <Switch {...field} checked={value} onCheckedChange={onChange} />
                        )}
                      </FormField>
                    </div>
                  </div>
                </Card>

                <RoutePreview pathValue={pathValue} targetValue={targetValue} includeSubpath={includeSubpath} />

                <Card className="block p-6">
                  <h3 className="mb-2 text-lg font-semibold text-foreground">{t("addPassThrough.headers.title")}</h3>
                  <p className="mb-6 text-sm text-muted-foreground">{t("addPassThrough.headers.description")}</p>

                  <FormField
                    control={form.control}
                    name="headers"
                    label={labelWithHint(t("addPassThrough.headers.label"), t("addPassThrough.headers.hint"))}
                    description={
                      <>
                        <span className="mb-1 block font-medium">{t("addPassThrough.headers.requiredNote")}</span>
                        <span className="block">{t("addPassThrough.headers.examples")}</span>
                      </>
                    }
                  >
                    {({ value, onChange }) => <KeyValueInput value={value} onChange={onChange} />}
                  </FormField>
                </Card>

                <Card className="block p-6">
                  <h3 className="mb-2 text-lg font-semibold text-foreground">
                    {t("addPassThrough.queryParams.title")}
                  </h3>
                  <p className="mb-6 text-sm text-muted-foreground">{t("addPassThrough.queryParams.description")}</p>

                  <FormField
                    control={form.control}
                    name="default_query_params"
                    label={labelWithHint(t("addPassThrough.queryParams.label"), t("addPassThrough.queryParams.hint"))}
                    description={
                      <>
                        <span className="mb-1 block font-medium">{t("addPassThrough.queryParams.methodsNote")}</span>
                        <span className="block">{t("addPassThrough.queryParams.overrideNote")}</span>
                      </>
                    }
                  >
                    {({ value, onChange }) => <QueryParamInput value={value} onChange={onChange} />}
                  </FormField>
                </Card>

                <FormField control={form.control} name="auth">
                  {({ value, onChange }) => <PassThroughSecuritySection authEnabled={value} onAuthChange={onChange} />}
                </FormField>

                <PassThroughGuardrailsSection accessToken={accessToken} value={guardrails} onChange={setGuardrails} />

                <Card className="block p-6">
                  <h3 className="mb-2 text-lg font-semibold text-foreground">
                    {t("addPassThrough.performance.title")}
                  </h3>
                  <p className="mb-6 text-sm text-muted-foreground">{t("addPassThrough.performance.description")}</p>

                  <FormField
                    control={form.control}
                    name="timeout"
                    label={labelWithHint(
                      t("addPassThrough.performance.timeoutLabel"),
                      t("addPassThrough.performance.timeoutHint"),
                    )}
                    description={t("addPassThrough.performance.timeoutDescription")}
                  >
                    {({ value, onChange, ref: _ref, ...field }) => (
                      <NumericalInput
                        {...field}
                        min={1}
                        step={1}
                        placeholder="600"
                        value={value ?? ""}
                        onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                          onChange(optionalText(event.target.value))
                        }
                      />
                    )}
                  </FormField>
                </Card>

                <Card className="block p-6">
                  <h3 className="mb-2 text-lg font-semibold text-foreground">{t("addPassThrough.billing.title")}</h3>
                  <p className="mb-6 text-sm text-muted-foreground">{t("addPassThrough.billing.description")}</p>

                  <FormField
                    control={form.control}
                    name="cost_per_request"
                    label={labelWithHint(t("addPassThrough.billing.costLabel"), t("addPassThrough.billing.costHint"))}
                    description={t("addPassThrough.billing.costDescription")}
                  >
                    {({ value, onChange, ref: _ref, ...field }) => (
                      <NumericalInput
                        {...field}
                        min={0}
                        step={0.001}
                        placeholder="2.0000"
                        value={value ?? ""}
                        onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                          onChange(optionalText(event.target.value))
                        }
                      />
                    )}
                  </FormField>
                </Card>

                <div className="flex items-center justify-end space-x-3 border-t border-border pt-6">
                  <Button type="button" variant="outline" onClick={handleCancel}>
                    {t("addPassThrough.cancel")}
                  </Button>
                  <Button type="submit" disabled={isLoading} aria-busy={isLoading}>
                    {isLoading && <UiLoadingSpinner className="size-4" />}
                    {isLoading ? t("addPassThrough.creating") : t("addPassThrough.submit")}
                  </Button>
                </div>
              </form>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
};

export default AddPassThroughEndpoint;

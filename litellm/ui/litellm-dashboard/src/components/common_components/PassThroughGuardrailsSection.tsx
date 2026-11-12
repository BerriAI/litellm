import React from "react";
import { CircleHelp, Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";

import GuardrailSelector from "../guardrails/GuardrailSelector";
import { TagsInput } from "@/app/(dashboard)/guardrails/_components/content_filter/TagsInput";
import { Field, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { codeSamples } from "@/i18n/untranslated/codeSamples";

type GuardrailFields = { request_fields?: string[]; response_fields?: string[] };
type GuardrailSettings = Record<string, GuardrailFields | null>;

interface PassThroughGuardrailsSectionProps {
  accessToken: string;
  value?: GuardrailSettings;
  onChange?: (guardrails: GuardrailSettings) => void;
  disabled?: boolean;
}

const labelWithHint = (label: React.ReactNode, hint: React.ReactNode): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const PassThroughGuardrailsSection: React.FC<PassThroughGuardrailsSectionProps> = ({
  accessToken,
  value = {},
  onChange,
  disabled = false,
}) => {
  const { t } = useTranslation();
  const selectedGuardrails = Object.keys(value);

  const emit = (next: GuardrailSettings) => {
    onChange?.(next);
  };

  const handleGuardrailChange = (guardrails: string[]) => {
    emit(Object.fromEntries(guardrails.map((name) => [name, value[name] ?? null])));
  };

  const handleFieldChange = (guardrailName: string, fieldType: keyof GuardrailFields, fields: string[]) => {
    const updated: GuardrailFields = {
      ...(value[guardrailName] ?? {}),
      [fieldType]: fields.length > 0 ? fields : undefined,
    };
    const isEmpty = !updated.request_fields && !updated.response_fields;
    emit({ ...value, [guardrailName]: isEmpty ? null : updated });
  };

  const appendField = (
    guardrailName: string,
    fieldType: keyof GuardrailFields,
    field: "query" | "documents[*]" | "results[*]",
  ) => {
    handleFieldChange(guardrailName, fieldType, [...(value[guardrailName]?.[fieldType] ?? []), field]);
  };

  return (
    <TooltipProvider>
      <Card className="block p-6">
        <h3 className="mb-2 text-lg font-semibold text-foreground">{t("passThroughGuardrailsSection.title")}</h3>
        <p className="mb-6 text-sm text-muted-foreground">{t("passThroughGuardrailsSection.description")}</p>

        <Alert variant="info" className="mb-4">
          <Info />
          <AlertTitle>
            {t("passThroughGuardrailsSection.fieldTargeting.title")}{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/pass_through_guardrails#field-level-targeting"
              target="_blank"
              rel="noopener noreferrer"
              className="text-info underline hover:text-info/80"
            >
              {t("passThroughGuardrailsSection.fieldTargeting.learnMore")}
            </a>
          </AlertTitle>
          <AlertDescription>
            <div className="space-y-2">
              <div>{t("passThroughGuardrailsSection.fieldTargeting.description")}</div>
              <div className="mt-2 space-y-1 text-xs">
                <div className="font-medium">{t("passThroughGuardrailsSection.fieldTargeting.commonExamples")}</div>
                <div>
                  • <code className="rounded-sm bg-muted px-1">query</code> -{" "}
                  {t("passThroughGuardrailsSection.fieldTargeting.singleField")}
                </div>
                <div>
                  • <code className="rounded-sm bg-muted px-1">documents[*].text</code> -{" "}
                  {t("passThroughGuardrailsSection.fieldTargeting.allDocumentText")}
                </div>
                <div>
                  • <code className="rounded-sm bg-muted px-1">messages[*].content</code> -{" "}
                  {t("passThroughGuardrailsSection.fieldTargeting.allMessageContents")}
                </div>
              </div>
            </div>
          </AlertDescription>
        </Alert>

        <Field>
          <FieldLabel htmlFor="pass-through-guardrails">
            {labelWithHint(
              t("passThroughGuardrailsSection.selectGuardrails"),
              t("passThroughGuardrailsSection.selectGuardrailsHint"),
            )}
          </FieldLabel>
          <GuardrailSelector
            accessToken={accessToken}
            value={selectedGuardrails}
            onChange={handleGuardrailChange}
            disabled={disabled}
          />
        </Field>

        {selectedGuardrails.length > 0 && (
          <div className="mt-6 space-y-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-medium text-foreground">
                {t("passThroughGuardrailsSection.fieldTargetingOptional")}
              </div>
              <div className="text-xs text-muted-foreground">{t("passThroughGuardrailsSection.tip")}</div>
            </div>
            {selectedGuardrails.map((guardrailName) => (
              <Card key={guardrailName} className="block bg-muted/50 p-4">
                <div className="mb-3 text-sm font-medium text-foreground">{guardrailName}</div>
                <div className="space-y-3">
                  <Field>
                    <div className="flex items-center justify-between">
                      <FieldLabel htmlFor={`${guardrailName}-request-fields`} className="text-xs text-muted-foreground">
                        {labelWithHint(
                          t("passThroughGuardrailsSection.requestFields.label"),
                          <div>
                            <div className="mb-1 font-medium">
                              {t("passThroughGuardrailsSection.requestFields.hint")}
                            </div>
                            <div className="space-y-1 text-xs">
                              <div>{t("passThroughGuardrailsSection.examples")}</div>
                              <div>{codeSamples.passThrough.guardrailRequestQuery}</div>
                              <div>{codeSamples.passThrough.guardrailRequestDocuments}</div>
                              <div>{codeSamples.passThrough.guardrailRequestMessages}</div>
                            </div>
                          </div>,
                        )}
                      </FieldLabel>
                      <div className="flex gap-1">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          disabled={disabled}
                          onClick={() => appendField(guardrailName, "request_fields", "query")}
                        >
                          {codeSamples.passThrough.guardrailAddQuery}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          disabled={disabled}
                          onClick={() => appendField(guardrailName, "request_fields", "documents[*]")}
                        >
                          {codeSamples.passThrough.guardrailAddDocuments}
                        </Button>
                      </div>
                    </div>
                    <TagsInput
                      id={`${guardrailName}-request-fields`}
                      placeholder={t("passThroughGuardrailsSection.requestFields.placeholder")}
                      value={value[guardrailName]?.request_fields ?? []}
                      onValueChange={(fields) => handleFieldChange(guardrailName, "request_fields", fields)}
                      tokenSeparators={[","]}
                      disabled={disabled}
                    />
                  </Field>
                  <Field>
                    <div className="flex items-center justify-between">
                      <FieldLabel
                        htmlFor={`${guardrailName}-response-fields`}
                        className="text-xs text-muted-foreground"
                      >
                        {labelWithHint(
                          t("passThroughGuardrailsSection.responseFields.label"),
                          <div>
                            <div className="mb-1 font-medium">
                              {t("passThroughGuardrailsSection.responseFields.hint")}
                            </div>
                            <div className="space-y-1 text-xs">
                              <div>{t("passThroughGuardrailsSection.examples")}</div>
                              <div>{codeSamples.passThrough.guardrailResponseResults}</div>
                              <div>{codeSamples.passThrough.guardrailResponseChoices}</div>
                            </div>
                          </div>,
                        )}
                      </FieldLabel>
                      <div className="flex gap-1">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          disabled={disabled}
                          onClick={() => appendField(guardrailName, "response_fields", "results[*]")}
                        >
                          {codeSamples.passThrough.guardrailAddResults}
                        </Button>
                      </div>
                    </div>
                    <TagsInput
                      id={`${guardrailName}-response-fields`}
                      placeholder={t("passThroughGuardrailsSection.responseFields.placeholder")}
                      value={value[guardrailName]?.response_fields ?? []}
                      onValueChange={(fields) => handleFieldChange(guardrailName, "response_fields", fields)}
                      tokenSeparators={[","]}
                      disabled={disabled}
                    />
                  </Field>
                </div>
              </Card>
            ))}
          </div>
        )}
      </Card>
    </TooltipProvider>
  );
};

export default PassThroughGuardrailsSection;

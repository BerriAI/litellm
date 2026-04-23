import React, { useState, useEffect } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetchUserModels } from "../organisms/create_key_button";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { tagInfoCall, tagUpdateCall } from "../networking";
import { Tag } from "./types";
import NotificationsManager from "../molecules/notifications_manager";
import NumericalInput from "../shared/numerical_input";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon, Info, X } from "lucide-react";
import { Controller, FormProvider, useForm } from "react-hook-form";

interface TagInfoViewProps {
  tagId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editTag: boolean;
}

interface TagEditValues {
  name: string;
  description: string;
  models: string[];
  max_budget: number | null;
  budget_duration: string | null;
}

function InfoTip({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="inline-block ml-1 h-3 w-3 text-muted-foreground cursor-help" />
        </TooltipTrigger>
        <TooltipContent>{children}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

const TagInfoView: React.FC<TagInfoViewProps> = ({
  tagId,
  onClose,
  accessToken,
  is_admin,
  editTag,
}) => {
  const form = useForm<TagEditValues>({
    defaultValues: {
      name: "",
      description: "",
      models: [],
      max_budget: null,
      budget_duration: null,
    },
    mode: "onSubmit",
  });
  const [tagDetails, setTagDetails] = useState<Tag | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editTag);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const copyToClipboard = async (
    text: string | null | undefined,
    key: string,
  ) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const fetchTagDetails = async () => {
    if (!accessToken) return;
    try {
      const response = await tagInfoCall(accessToken, [tagId]);
      const tagData = response[tagId];
      if (tagData) {
        setTagDetails(tagData);
        if (editTag) {
          form.reset({
            name: tagData.name,
            description: tagData.description ?? "",
            models: tagData.models ?? [],
            max_budget: tagData.litellm_budget_table?.max_budget ?? null,
            budget_duration:
              tagData.litellm_budget_table?.budget_duration ?? null,
          });
        }
      }
    } catch (error) {
      console.error("Error fetching tag details:", error);
      NotificationsManager.fromBackend("Error fetching tag details: " + error);
    }
  };

  useEffect(() => {
    fetchTagDetails();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tagId, accessToken]);

  useEffect(() => {
    if (accessToken) {
      fetchUserModels("dummy-user", "Admin", accessToken, setUserModels);
    }
  }, [accessToken]);

  const handleSave = form.handleSubmit(async (values) => {
    if (!accessToken) return;
    try {
      await tagUpdateCall(accessToken, {
        name: values.name,
        description: values.description,
        models: values.models,
        max_budget: values.max_budget ?? undefined,
        budget_duration: values.budget_duration ?? undefined,
      });
      NotificationsManager.success("Tag updated successfully");
      setIsEditing(false);
      fetchTagDetails();
    } catch (error) {
      console.error("Error updating tag:", error);
      NotificationsManager.fromBackend("Error updating tag: " + error);
    }
  });

  if (!tagDetails) return <div>Loading...</div>;

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} variant="outline" className="mb-4">
            ← Back to Tags
          </Button>
          <div className="flex items-center gap-2">
            <span className="font-medium">Tag Name:</span>
            <span className="font-mono px-2 py-1 bg-muted rounded text-sm border border-border">
              {tagDetails.name}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => copyToClipboard(tagDetails.name, "tag-name")}
              aria-label="Copy tag name"
              className="h-7 w-7"
            >
              {copiedStates["tag-name"] ? (
                <CheckIcon size={12} />
              ) : (
                <CopyIcon size={12} />
              )}
            </Button>
          </div>
          <p className="text-muted-foreground text-sm">
            {tagDetails.description || "No description"}
          </p>
        </div>
        {is_admin && !isEditing && (
          <Button onClick={() => setIsEditing(true)}>Edit Tag</Button>
        )}
      </div>

      {isEditing ? (
        <Card className="p-6">
          <FormProvider {...form}>
            <form onSubmit={handleSave} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">
                  Tag Name <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="name"
                  {...form.register("name", {
                    required: "Please input a tag name",
                  })}
                />
                {form.formState.errors.name && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.name.message as string}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  rows={4}
                  {...form.register("description")}
                />
              </div>

              <div className="space-y-2">
                <Label>
                  Allowed Models
                  <InfoTip>
                    Select which models are allowed to process this type of data
                  </InfoTip>
                </Label>
                <Controller
                  control={form.control}
                  name="models"
                  render={({ field }) => {
                    const remaining = userModels.filter(
                      (m) => !field.value.includes(m),
                    );
                    return (
                      <div className="space-y-2">
                        <Select
                          value=""
                          onValueChange={(v) => {
                            if (v) field.onChange([...field.value, v]);
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select Models" />
                          </SelectTrigger>
                          <SelectContent>
                            {remaining.length === 0 ? (
                              <div className="py-2 px-3 text-sm text-muted-foreground">
                                No more models available
                              </div>
                            ) : (
                              remaining.map((modelId) => (
                                <SelectItem key={modelId} value={modelId}>
                                  {getModelDisplayName(modelId)}
                                </SelectItem>
                              ))
                            )}
                          </SelectContent>
                        </Select>
                        {field.value.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {field.value.map((id) => (
                              <Badge
                                key={id}
                                variant="secondary"
                                className="gap-1"
                              >
                                {getModelDisplayName(id)}
                                <button
                                  type="button"
                                  onClick={() =>
                                    field.onChange(
                                      field.value.filter((v) => v !== id),
                                    )
                                  }
                                  aria-label={`Remove ${id}`}
                                >
                                  <X size={10} />
                                </button>
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  }}
                />
              </div>

              <Accordion type="single" collapsible className="mt-4 mb-4">
                <AccordionItem value="budget">
                  <AccordionTrigger className="font-semibold">
                    Budget &amp; Rate Limits
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label>
                          Max Budget (USD)
                          <InfoTip>
                            Maximum amount in USD this tag can spend
                          </InfoTip>
                        </Label>
                        <Controller
                          control={form.control}
                          name="max_budget"
                          render={({ field }) => (
                            <NumericalInput
                              step={0.01}
                              precision={2}
                              style={{ width: "100%" }}
                              value={field.value ?? ""}
                              onChange={(v: number | null | undefined) =>
                                field.onChange(v ?? null)
                              }
                            />
                          )}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>
                          Reset Budget
                          <InfoTip>
                            How often the budget should reset
                          </InfoTip>
                        </Label>
                        <Controller
                          control={form.control}
                          name="budget_duration"
                          render={({ field }) => (
                            <BudgetDurationDropdown
                              value={field.value ?? undefined}
                              onChange={(value) =>
                                field.onChange((value as string) ?? null)
                              }
                            />
                          )}
                        />
                      </div>
                      <div className="mt-4 p-3 bg-muted rounded-md border border-border">
                        <p className="text-sm text-muted-foreground">
                          TPM/RPM limits for tags are not currently supported.
                          If you need this feature, please{" "}
                          <a
                            href="https://github.com/BerriAI/litellm/issues/new"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-primary hover:underline"
                          >
                            create a GitHub issue
                          </a>
                          .
                        </p>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>

              <div className="flex justify-end space-x-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setIsEditing(false)}
                >
                  Cancel
                </Button>
                <Button type="submit">Save Changes</Button>
              </div>
            </form>
          </FormProvider>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold">Tag Details</h3>
            <div className="space-y-4 mt-4 text-sm">
              <div>
                <p className="font-medium">Name</p>
                <p>{tagDetails.name}</p>
              </div>
              <div>
                <p className="font-medium">Description</p>
                <p>{tagDetails.description || "-"}</p>
              </div>
              <div>
                <p className="font-medium">Allowed Models</p>
                <div className="flex flex-wrap gap-2 mt-2">
                  {!tagDetails.models || tagDetails.models.length === 0 ? (
                    <Badge variant="destructive">All Models</Badge>
                  ) : (
                    tagDetails.models.map((modelId) => (
                      <TooltipProvider key={modelId}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Badge variant="secondary">
                              {tagDetails.model_info?.[modelId] || modelId}
                            </Badge>
                          </TooltipTrigger>
                          <TooltipContent>{`ID: ${modelId}`}</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ))
                  )}
                </div>
              </div>
              <div>
                <p className="font-medium">Created</p>
                <p>
                  {tagDetails.created_at
                    ? new Date(tagDetails.created_at).toLocaleString()
                    : "-"}
                </p>
              </div>
              <div>
                <p className="font-medium">Last Updated</p>
                <p>
                  {tagDetails.updated_at
                    ? new Date(tagDetails.updated_at).toLocaleString()
                    : "-"}
                </p>
              </div>
            </div>
          </Card>

          {tagDetails.litellm_budget_table && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold">Budget &amp; Rate Limits</h3>
              <div className="space-y-4 mt-4 text-sm">
                {tagDetails.litellm_budget_table.max_budget !== undefined &&
                  tagDetails.litellm_budget_table.max_budget !== null && (
                    <div>
                      <p className="font-medium">Max Budget</p>
                      <p>${tagDetails.litellm_budget_table.max_budget}</p>
                    </div>
                  )}
                {tagDetails.litellm_budget_table.budget_duration && (
                  <div>
                    <p className="font-medium">Budget Duration</p>
                    <p>{tagDetails.litellm_budget_table.budget_duration}</p>
                  </div>
                )}
                {tagDetails.litellm_budget_table.tpm_limit !== undefined &&
                  tagDetails.litellm_budget_table.tpm_limit !== null && (
                    <div>
                      <p className="font-medium">TPM Limit</p>
                      <p>
                        {tagDetails.litellm_budget_table.tpm_limit.toLocaleString()}
                      </p>
                    </div>
                  )}
                {tagDetails.litellm_budget_table.rpm_limit !== undefined &&
                  tagDetails.litellm_budget_table.rpm_limit !== null && (
                    <div>
                      <p className="font-medium">RPM Limit</p>
                      <p>
                        {tagDetails.litellm_budget_table.rpm_limit.toLocaleString()}
                      </p>
                    </div>
                  )}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};

export default TagInfoView;

import React from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { Info, X } from "lucide-react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import BudgetDurationDropdown from "../../common_components/budget_duration_dropdown";
import NumericalInput from "../../shared/numerical_input";

interface ModelInfo {
  model_name: string;
  litellm_params: { model: string };
  model_info: { id: string };
}

interface CreateTagFormValues {
  tag_name: string;
  description: string;
  allowed_llms: string[];
  max_budget: number | null;
  budget_duration: string | null;
}

const defaultValues: CreateTagFormValues = {
  tag_name: "",
  description: "",
  allowed_llms: [],
  max_budget: null,
  budget_duration: null,
};

interface CreateTagModalProps {
  visible: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onSubmit: (values: any) => void;
  availableModels: ModelInfo[];
}

function InfoTip({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="h-3 w-3 text-muted-foreground inline-block ml-1 cursor-help" />
        </TooltipTrigger>
        <TooltipContent>{children}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

const CreateTagModal: React.FC<CreateTagModalProps> = ({
  visible,
  onCancel,
  onSubmit,
  availableModels,
}) => {
  const form = useForm<CreateTagFormValues>({
    defaultValues,
    mode: "onSubmit",
  });

  const handleFinish = form.handleSubmit((values) => {
    onSubmit(values);
    form.reset(defaultValues);
  });

  const handleCancel = () => {
    form.reset(defaultValues);
    onCancel();
  };

  // Multi-select scaffolding (chips + add via Select).
  const allowedLlms = form.watch("allowed_llms");
  const remainingModels = availableModels.filter(
    (m) => !allowedLlms.includes(m.model_info.id),
  );

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Create New Tag</DialogTitle>
          <DialogDescription className="sr-only">
            Define a new tag with allowed models and optional budget / rate
            limits.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleFinish} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="tag_name">
                Tag Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="tag_name"
                {...form.register("tag_name", {
                  required: "Please input a tag name",
                })}
              />
              {form.formState.errors.tag_name && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.tag_name.message as string}
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
                  Select which models are allowed to process requests from
                  this tag
                </InfoTip>
              </Label>
              <Controller
                control={form.control}
                name="allowed_llms"
                render={({ field }) => (
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
                        {remainingModels.length === 0 ? (
                          <div className="py-2 px-3 text-sm text-muted-foreground">
                            No more models available
                          </div>
                        ) : (
                          remainingModels.map((model) => (
                            <SelectItem
                              key={model.model_info.id}
                              value={model.model_info.id}
                            >
                              {model.model_name}{" "}
                              <span className="text-muted-foreground text-xs ml-2">
                                ({model.model_info.id})
                              </span>
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                    {field.value.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {field.value.map((id) => {
                          const m = availableModels.find(
                            (mm) => mm.model_info.id === id,
                          );
                          return (
                            <span
                              key={id}
                              className="inline-flex items-center gap-1 rounded bg-secondary text-secondary-foreground px-2 py-0.5 text-xs"
                            >
                              {m?.model_name ?? id}
                              <button
                                type="button"
                                onClick={() =>
                                  field.onChange(
                                    field.value.filter((v) => v !== id),
                                  )
                                }
                                aria-label={`Remove ${m?.model_name ?? id}`}
                              >
                                <X size={10} />
                              </button>
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              />
            </div>

            <Accordion type="single" collapsible className="mt-4 mb-4">
              <AccordionItem value="budget">
                <AccordionTrigger className="font-semibold">
                  Budget &amp; Rate Limits (Optional)
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="max_budget">
                        Max Budget (USD)
                        <InfoTip>
                          Maximum amount in USD this tag can spend. When
                          reached, requests with this tag will be blocked.
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
                            value={field.value ?? undefined}
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
                          How often the budget should reset. For example,
                          &quot;daily&quot; resets every 24 hours.
                        </InfoTip>
                      </Label>
                      <Controller
                        control={form.control}
                        name="budget_duration"
                        render={({ field }) => (
                          <BudgetDurationDropdown
                            onChange={(value) =>
                              field.onChange(value as string | null)
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

            <DialogFooter>
              <Button type="button" variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
              <Button type="submit">Create Tag</Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default CreateTagModal;

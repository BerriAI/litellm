import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { UploadProps } from "../add_model/add_model_upload_types";
import React, { useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { Providers, providerLogoMap } from "../provider_info_helpers";

interface AddCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onAddCredential: (values: any) => void;
  uploadProps: UploadProps;
}

type AddCredentialFormValues = {
  credential_name: string;
  custom_llm_provider: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
};

const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({
  open,
  onCancel,
  onAddCredential,
  uploadProps,
}) => {
  const form = useForm<AddCredentialFormValues>({
    defaultValues: {
      credential_name: "",
      custom_llm_provider: "",
    },
    mode: "onSubmit",
  });
  const [selectedProvider, setSelectedProvider] = useState<Providers>(
    Providers.OpenAI,
  );

  const handleSubmit = form.handleSubmit((values) => {
    const filteredValues = Object.entries(values).reduce(
      (acc, [key, value]) => {
        if (value !== "" && value !== undefined && value !== null) {
          acc[key] = value;
        }
        return acc;
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      {} as any,
    );
    onAddCredential(filteredValues);
    form.reset({ credential_name: "", custom_llm_provider: "" });
  });

  const handleCancel = () => {
    onCancel();
    form.reset({ credential_name: "", custom_llm_provider: "" });
  };

  return (
    <Dialog open={open} onOpenChange={(o) => (!o ? handleCancel() : undefined)}>
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add New Credential</DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleSubmit}>
            <div className="space-y-4 mb-4">
              <div className="space-y-2">
                <div className="flex items-center">
                  <Label htmlFor="credential_name">Credential Name:</Label>
                  <span aria-hidden="true" className="text-destructive ml-1">
                    *
                  </span>
                </div>
                <Input
                  id="credential_name"
                  placeholder="Enter a friendly name for these credentials"
                  {...form.register("credential_name", {
                    required: "Credential name is required",
                  })}
                />
                {form.formState.errors.credential_name && (
                  <p className="text-sm text-destructive">
                    {String(form.formState.errors.credential_name.message)}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex items-center">
                  <Label
                    htmlFor="custom_llm_provider"
                    title="Helper to auto-populate provider specific fields"
                  >
                    Provider:
                  </Label>
                  <span aria-hidden="true" className="text-destructive ml-1">
                    *
                  </span>
                </div>
                <Controller
                  control={form.control}
                  name="custom_llm_provider"
                  rules={{ required: "Required" }}
                  render={({ field }) => (
                    <Select
                      value={field.value || ""}
                      onValueChange={(v) => {
                        field.onChange(v);
                        setSelectedProvider(v as Providers);
                      }}
                    >
                      <SelectTrigger id="custom_llm_provider">
                        <SelectValue placeholder="Select a provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(Providers).map(
                          ([providerEnum, providerDisplayName]) => (
                            <SelectItem key={providerEnum} value={providerEnum}>
                              <div className="flex items-center space-x-2">
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                  src={providerLogoMap[providerDisplayName]}
                                  alt={`${providerEnum} logo`}
                                  className="w-5 h-5"
                                  onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    const parent = target.parentElement;
                                    if (parent) {
                                      const fallbackDiv =
                                        document.createElement("div");
                                      fallbackDiv.className =
                                        "w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs";
                                      fallbackDiv.textContent =
                                        providerDisplayName.charAt(0);
                                      parent.replaceChild(fallbackDiv, target);
                                    }
                                  }}
                                />
                                <span>{providerDisplayName}</span>
                              </div>
                            </SelectItem>
                          ),
                        )}
                      </SelectContent>
                    </Select>
                  )}
                />
                {form.formState.errors.custom_llm_provider && (
                  <p className="text-sm text-destructive">
                    {String(form.formState.errors.custom_llm_provider.message)}
                  </p>
                )}
              </div>

              <ProviderSpecificFields
                selectedProvider={selectedProvider}
                uploadProps={uploadProps}
              />
            </div>

            <div className="flex justify-between items-center">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <a
                      href="https://github.com/BerriAI/litellm/issues"
                      className="text-primary hover:underline"
                    >
                      Need Help?
                    </a>
                  </TooltipTrigger>
                  <TooltipContent>Get help on our github</TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <div className="flex gap-2">
                <Button variant="outline" type="button" onClick={handleCancel}>
                  Cancel
                </Button>
                <Button type="submit">Add Credential</Button>
              </div>
            </div>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddCredentialsModal;

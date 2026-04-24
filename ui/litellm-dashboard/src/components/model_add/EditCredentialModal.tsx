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
import type { UploadProps } from "antd/es/upload";
import { useEffect, useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { CredentialItem } from "../networking";
import { Providers, providerLogoMap } from "../provider_info_helpers";

interface EditCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onUpdateCredential: (values: any) => void;
  uploadProps: UploadProps;
  existingCredential: CredentialItem | null;
}

type EditCredentialFormValues = {
  credential_name: string;
  custom_llm_provider: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
};

export default function EditCredentialsModal({
  open,
  onCancel,
  onUpdateCredential,
  uploadProps,
  existingCredential,
}: EditCredentialsModalProps) {
  const form = useForm<EditCredentialFormValues>({
    defaultValues: {
      credential_name: existingCredential?.credential_name ?? "",
      custom_llm_provider: "",
    },
    mode: "onSubmit",
  });
  const [selectedProvider, setSelectedProvider] = useState<Providers>(
    Providers.Anthropic,
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
    onUpdateCredential(filteredValues);
    form.reset({ credential_name: "", custom_llm_provider: "" });
  });

  useEffect(() => {
    if (existingCredential) {
      const credentialValues = Object.entries(
        existingCredential.credential_values || {},
      ).reduce(
        (acc, [key, value]) => {
          acc[key] = value ?? null;
          return acc;
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        {} as Record<string, any>,
      );

      form.reset({
        credential_name: existingCredential.credential_name,
        custom_llm_provider:
          existingCredential.credential_info.custom_llm_provider,
        ...credentialValues,
      });
      setSelectedProvider(
        existingCredential.credential_info.custom_llm_provider as Providers,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [existingCredential]);

  const handleCancel = () => {
    onCancel();
    form.reset({ credential_name: "", custom_llm_provider: "" });
  };

  const isNameLocked = Boolean(existingCredential?.credential_name);

  return (
    <Dialog open={open} onOpenChange={(o) => (!o ? handleCancel() : undefined)}>
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Edit Credential</DialogTitle>
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
                  disabled={isNameLocked}
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
                <Button type="submit">Update Credential</Button>
              </div>
            </div>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}

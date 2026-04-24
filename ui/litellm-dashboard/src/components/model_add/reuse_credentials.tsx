import React, { useEffect } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { CredentialItem } from "../networking";

interface ReuseCredentialsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onAddCredential: (values: any) => void;
  existingCredential: CredentialItem | null;
  setIsCredentialModalOpen: (isVisible: boolean) => void;
}

type ReuseCredentialFormValues = {
  credential_name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
};

const ReuseCredentialsModal: React.FC<ReuseCredentialsModalProps> = ({
  isVisible,
  onCancel,
  onAddCredential,
  existingCredential,
  setIsCredentialModalOpen,
}) => {
  const credentialValueEntries = Object.entries(
    existingCredential?.credential_values || {},
  );

  const form = useForm<ReuseCredentialFormValues>({
    defaultValues: {
      credential_name: existingCredential?.credential_name ?? "",
      ...credentialValueEntries.reduce(
        (acc, [key, value]) => {
          acc[key] = (value as string | number | null | undefined) ?? "";
          return acc;
        },
        {} as Record<string, string | number | null | undefined>,
      ),
    },
  });

  useEffect(() => {
    form.reset({
      credential_name: existingCredential?.credential_name ?? "",
      ...credentialValueEntries.reduce(
        (acc, [key, value]) => {
          acc[key] = (value as string | number | null | undefined) ?? "";
          return acc;
        },
        {} as Record<string, string | number | null | undefined>,
      ),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [existingCredential]);

  const handleSubmit = form.handleSubmit((values) => {
    onAddCredential(values);
    form.reset();
    setIsCredentialModalOpen(false);
  });

  const handleCancel = () => {
    onCancel();
    form.reset();
  };

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => {
        if (!o) {
          handleCancel();
        }
      }}
    >
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Reuse Credentials</DialogTitle>
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

              {credentialValueEntries.map(([key]) => (
                <div key={key} className="space-y-2">
                  <Label htmlFor={`reuse-${key}`}>{key}</Label>
                  <Input
                    id={`reuse-${key}`}
                    placeholder={`Enter ${key}`}
                    disabled
                    {...form.register(key)}
                  />
                </div>
              ))}
            </div>

            <div className="flex justify-between items-center">
              <a
                href="https://github.com/BerriAI/litellm/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:text-primary/80 text-sm"
                title="Get help on our github"
              >
                Need Help?
              </a>

              <div className="flex gap-2">
                <Button type="button" variant="outline" onClick={handleCancel}>
                  Cancel
                </Button>
                <Button type="submit">Reuse Credentials</Button>
              </div>
            </div>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default ReuseCredentialsModal;

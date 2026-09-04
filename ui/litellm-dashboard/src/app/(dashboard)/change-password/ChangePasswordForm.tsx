"use client";

import React, { useState } from "react";
import { CircleAlert } from "lucide-react";
import { z } from "zod/v4";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FieldGroup } from "@/components/ui/field";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { changePasswordCall } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

const changePasswordSchema = z
  .object({
    currentPassword: z.string().min(1, "Current password is required"),
    newPassword: z.string().min(1, "New password is required"),
    confirmNewPassword: z.string().min(1, "Confirm your new password"),
  })
  .refine((values) => values.newPassword === values.confirmNewPassword, {
    message: "New passwords do not match",
    path: ["confirmNewPassword"],
  });

type ChangePasswordValues = z.infer<typeof changePasswordSchema>;

export function ChangePasswordForm() {
  const { accessToken } = useAuthorized();
  const form = useZodForm(changePasswordSchema, {
    defaultValues: { currentPassword: "", newPassword: "", confirmNewPassword: "" },
  });
  const [isPending, setIsPending] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = async (values: ChangePasswordValues) => {
    if (!accessToken) return;
    setSubmitError(null);
    setIsPending(true);
    try {
      await changePasswordCall(accessToken, values.currentPassword, values.newPassword);
      toast.success("Password updated");
      form.reset();
    } catch (error) {
      setSubmitError(extractProxyErrorMessage(error));
    } finally {
      setIsPending(false);
    }
  };

  return (
    <div className="mx-auto mt-10 w-full max-w-md">
      <Card>
        <CardContent>
          <h3 className="text-2xl font-semibold text-foreground">Change Password</h3>
          <p className="text-sm text-muted-foreground">
            Enter your current password and choose a new one. The new password must meet this proxy&apos;s password
            policy.
          </p>

          <form className="mb-2 mt-8" onSubmit={form.handleSubmit(handleSubmit)}>
            <FieldGroup>
              <FormField control={form.control} name="currentPassword" label="Current Password">
                {({ ref, ...field }) => <PasswordInput {...field} ref={ref} autoComplete="current-password" />}
              </FormField>

              <FormField control={form.control} name="newPassword" label="New Password">
                {({ ref, ...field }) => <PasswordInput {...field} ref={ref} autoComplete="new-password" />}
              </FormField>

              <FormField control={form.control} name="confirmNewPassword" label="Confirm New Password">
                {({ ref, ...field }) => <PasswordInput {...field} ref={ref} autoComplete="new-password" />}
              </FormField>
            </FieldGroup>

            {submitError && (
              <Alert variant="error" className="mt-6">
                <CircleAlert />
                <AlertTitle>{submitError}</AlertTitle>
              </Alert>
            )}

            <div className="mt-8">
              <Button type="submit" disabled={isPending}>
                {isPending && <UiLoadingSpinner className="size-4" role="img" aria-label="loading" />}
                Change Password
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

export default ChangePasswordForm;

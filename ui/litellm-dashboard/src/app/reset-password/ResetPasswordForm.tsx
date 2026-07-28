"use client";

import React from "react";
import { z } from "zod/v4";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useValidateResetToken, useResetPassword } from "@/app/(dashboard)/hooks/passwordReset/usePasswordReset";
import { useZodForm } from "@/lib/forms/useZodForm";
import { FormField } from "@/components/shared/form/FormField";
import { FieldGroup } from "@/components/shared/form/field";
import { getLoginUrl } from "@/utils/returnUrlUtils";

const resetPasswordSchema = z
  .object({
    password: z.string().min(1, "Please enter a new password"),
    confirm_password: z.string().min(1, "Please confirm your new password"),
  })
  .refine((data) => data.password === data.confirm_password, {
    path: ["confirm_password"],
    message: "Passwords do not match",
  });

type ResetPasswordFormProps = {
  token: string | null;
};

function ResetPasswordCardHeader({ description }: { description?: React.ReactNode }) {
  return (
    <CardHeader>
      <CardTitle className="text-center">🚅 LiteLLM</CardTitle>
      <h1 className="text-lg font-semibold">Reset Password</h1>
      {description && <p className="text-sm text-muted-foreground">{description}</p>}
    </CardHeader>
  );
}

export function ResetPasswordForm({ token }: ResetPasswordFormProps) {
  const { data: validationData, isLoading: isValidating, isError: isValidationError } = useValidateResetToken(token);
  const { mutate: submitResetPassword, isPending, isSuccess, error: resetError } = useResetPassword();
  const form = useZodForm(resetPasswordSchema, { defaultValues: { password: "", confirm_password: "" } });

  if (!token || isValidationError) {
    return (
      <div className="mx-auto w-full max-w-md mt-10">
        <Card>
          <ResetPasswordCardHeader />
          <CardContent>
            <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-red-700">
              This link is invalid or has expired.
            </div>
            <div className="mt-4">
              <a href="/ui/forgot-password" className="text-sm text-primary underline-offset-4 hover:underline">
                Request a new link
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isValidating) {
    return (
      <div className="mx-auto w-full max-w-md mt-10">
        <Card>
          <ResetPasswordCardHeader />
          <CardContent>
            <div className="flex justify-center">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isSuccess) {
    return (
      <div className="mx-auto w-full max-w-md mt-10">
        <Card>
          <ResetPasswordCardHeader />
          <CardContent>
            <div className="rounded-md border border-green-200 bg-green-50 px-4 py-2 text-green-700">
              Password reset successfully.
            </div>
            <div className="mt-4">
              <a href={getLoginUrl()} className="text-sm text-primary underline-offset-4 hover:underline">
                Back to Login
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const onSubmit = form.handleSubmit((values) => {
    if (!token) return;
    submitResetPassword({ token, newPassword: values.password });
  });

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card>
        <ResetPasswordCardHeader description={`Resetting password for ${validationData?.user_email}`} />
        <CardContent>
          <form className="flex flex-col gap-4" onSubmit={onSubmit}>
            <FieldGroup>
              <FormField control={form.control} name="password" label="New Password">
                {({ ref, ...field }) => <Input {...field} ref={ref} type="password" />}
              </FormField>
              <FormField control={form.control} name="confirm_password" label="Confirm New Password">
                {({ ref, ...field }) => <Input {...field} ref={ref} type="password" />}
              </FormField>
            </FieldGroup>

            {resetError && (
              <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-red-700">
                {(resetError as Error).message}
              </div>
            )}

            <div className="mt-6">
              <Button type="submit" disabled={isPending}>
                {isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
                Reset Password
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

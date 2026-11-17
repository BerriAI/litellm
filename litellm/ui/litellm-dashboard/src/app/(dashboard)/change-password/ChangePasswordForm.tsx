"use client";

import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { CircleAlert } from "lucide-react";
import { z } from "zod";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FieldGroup } from "@/components/ui/field";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { changePasswordCall, getProxyBaseUrl, getUiPageUrl } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import { revokeSessionAndClearClientState } from "@/app/(dashboard)/hooks/useLogout";
import { getLoginUrl } from "@/utils/returnUrlUtils";
import i18n from "@/i18n/i18n";

const changePasswordSchema = z
  .object({
    currentPassword: z.string().min(1, { error: () => i18n.t("changePassword.schema.currentRequired") }),
    newPassword: z.string().min(1, { error: () => i18n.t("changePassword.schema.newRequired") }),
    confirmNewPassword: z.string().min(1, { error: () => i18n.t("changePassword.schema.confirmRequired") }),
  })
  .refine((values) => values.newPassword === values.confirmNewPassword, {
    error: () => i18n.t("changePassword.schema.mismatch"),
    path: ["confirmNewPassword"],
  });

type ChangePasswordValues = z.infer<typeof changePasswordSchema>;

export function ChangePasswordForm() {
  const { t } = useTranslation();
  const { accessToken, passwordResetRequired } = useAuthorized();
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
      if (passwordResetRequired) {
        // The session key was minted restricted; only a fresh login lifts it.
        // Revoke it server-side too (best-effort) so it doesn't sit valid
        // until the expiry reaper gets to it.
        toast.success(t("changePassword.form.toasts.updatedSignIn"));
        await revokeSessionAndClearClientState(accessToken);
        window.location.replace(getUiPageUrl(getLoginUrl(getProxyBaseUrl())));
        return;
      }
      toast.success(t("changePassword.form.toasts.updated"));
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
          <h3 className="text-2xl font-semibold text-foreground">{t("changePassword.form.title")}</h3>
          <p className="text-sm text-muted-foreground">{t("changePassword.form.description")}</p>

          {passwordResetRequired && (
            <Alert variant="warning" className="mt-4">
              <CircleAlert />
              <AlertTitle>{t("changePassword.form.resetRequired")}</AlertTitle>
            </Alert>
          )}

          <form className="mb-2 mt-8" onSubmit={form.handleSubmit(handleSubmit)}>
            <FieldGroup>
              <FormField control={form.control} name="currentPassword" label={t("changePassword.form.currentPassword")}>
                {({ ref, ...field }) => <PasswordInput {...field} ref={ref} autoComplete="current-password" />}
              </FormField>

              <FormField control={form.control} name="newPassword" label={t("changePassword.form.newPassword")}>
                {({ ref, ...field }) => <PasswordInput {...field} ref={ref} autoComplete="new-password" />}
              </FormField>

              <FormField
                control={form.control}
                name="confirmNewPassword"
                label={t("changePassword.form.confirmNewPassword")}
              >
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
                {isPending && (
                  <UiLoadingSpinner className="size-4" role="img" aria-label={t("changePassword.form.loading")} />
                )}
                {t("changePassword.form.submit")}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

export default ChangePasswordForm;

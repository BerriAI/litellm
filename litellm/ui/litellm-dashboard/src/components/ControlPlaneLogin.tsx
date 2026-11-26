"use client";

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { controlPlaneApi } from "@/lib/http/controlPlane";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ErrorNotice } from "./ControlPlaneShared";
import { LoginLogo } from "@/app/login/LoginPage";

export function ControlPlaneLogin({
  prefix,
  expired,
  onLogin,
}: {
  prefix: string;
  expired: boolean;
  onLogin: (token: string) => void;
}) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>();

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setError(undefined);
    const fields = new FormData(event.currentTarget);
    try {
      const response = await controlPlaneApi(prefix).login(
        String(fields.get("username") ?? ""),
        String(fields.get("password") ?? ""),
      );
      onLogin(response.access_token);
    } catch (failure) {
      setError(failure);
    } finally {
      setPending(false);
    }
  }
  return (
    <main className="flex min-h-screen items-center justify-center bg-muted p-4">
      <Card className="w-full max-w-lg shadow-md">
        <CardContent className="flex flex-col gap-4">
          <LoginLogo />
          <div className="text-center">
            <h1 className="text-2xl font-semibold text-foreground">{t("controlPlane.controlPlaneLogin")}</h1>
          </div>
          <form className="grid gap-4" onSubmit={submit}>
            <label className="grid gap-1 text-sm font-medium">
              {t("login.form.username.label")}
              <Input name="username" className="h-10" autoComplete="username" required />
            </label>
            <label className="grid gap-1 text-sm font-medium">
              {t("login.form.password.label")}
              <Input name="password" type="password" className="h-10" autoComplete="current-password" required />
            </label>
            <ErrorNotice
              error={error ?? (expired ? new Error("Your control plane session expired. Sign in again.") : undefined)}
            />
            <Button type="submit" size="lg" disabled={pending}>
              {t(pending ? "controlPlane.signingIn" : "controlPlane.signIn")}
            </Button>
          </form>
        </CardContent>
      </Card>
    </main>
  );
}

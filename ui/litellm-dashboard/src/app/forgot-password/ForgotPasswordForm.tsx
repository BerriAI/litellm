"use client";

import React, { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useForgotPassword } from "@/app/(dashboard)/hooks/passwordReset/usePasswordReset";
import { getLoginUrl } from "@/utils/returnUrlUtils";

export function ForgotPasswordForm() {
  const [email, setEmail] = useState("");
  const { mutate: submitForgotPassword, isPending, isSuccess, error } = useForgotPassword();

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    submitForgotPassword(email);
  };

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card>
        <CardHeader>
          <CardTitle className="text-center">🚅 LiteLLM</CardTitle>
          <h1 className="text-lg font-semibold">Forgot Password</h1>
          <p className="text-sm text-muted-foreground">
            Enter your email address and we will send you a link to reset your password.
          </p>
        </CardHeader>
        <CardContent>
          {isSuccess ? (
            <div className="rounded-md border border-green-200 bg-green-50 px-4 py-2 text-green-700">
              If an account exists for this email, a password reset link has been sent.
            </div>
          ) : (
            <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="email">Email Address</Label>
                <Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
              </div>

              {error && (
                <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-red-700">
                  {(error as Error).message}
                </div>
              )}

              <div className="mt-6">
                <Button type="submit" disabled={isPending}>
                  {isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
                  Send Reset Link
                </Button>
              </div>
            </form>
          )}

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

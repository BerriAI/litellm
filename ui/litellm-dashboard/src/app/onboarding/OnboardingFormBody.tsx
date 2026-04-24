import React from "react";
import { useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Info, LoaderCircle } from "lucide-react";

type OnboardingFormBodyProps = {
  variant: "signup" | "reset_password";
  userEmail: string;
  isPending: boolean;
  claimError: string | null;
  onSubmit: (values: { password: string }) => void;
};

type FormValues = {
  user_email: string;
  password: string;
};

export function OnboardingFormBody({
  variant,
  userEmail,
  isPending,
  claimError,
  onSubmit,
}: OnboardingFormBodyProps) {
  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
  } = useForm<FormValues>({
    defaultValues: { user_email: userEmail, password: "" },
  });

  React.useEffect(() => {
    if (userEmail) setValue("user_email", userEmail);
  }, [userEmail, setValue]);

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card className="p-6">
        <h5 className="text-center mb-5 text-base font-semibold">
          🚅 LiteLLM
        </h5>
        <h3 className="text-xl font-semibold">
          {variant === "reset_password" ? "Reset Password" : "Sign Up"}
        </h3>
        <p className="text-sm mt-1">
          {variant === "reset_password"
            ? "Reset your password to access Admin UI."
            : "Claim your user account to login to Admin UI."}
        </p>

        {variant === "signup" && (
          <div className="mt-4 flex gap-2 items-start p-3 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 text-blue-800 dark:text-blue-200">
            <Info className="h-4 w-4 mt-0.5 shrink-0" />
            <div className="flex-1">
              <div className="font-semibold">SSO</div>
              <div className="flex justify-between items-center text-sm">
                <span>SSO is under the Enterprise Tier.</span>
                <Button asChild size="sm">
                  <a
                    href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Get Free Trial
                  </a>
                </Button>
              </div>
            </div>
          </div>
        )}

        <form
          className="mt-10 mb-5 flex flex-col gap-4"
          onSubmit={handleSubmit((values) =>
            onSubmit({ password: values.password }),
          )}
        >
          <div className="space-y-2">
            <Label htmlFor="user_email">Email Address</Label>
            <Input
              id="user_email"
              type="email"
              disabled
              {...register("user_email")}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              {...register("password", {
                required: "password required to sign up",
              })}
            />
            {errors.password ? (
              <p className="text-sm text-destructive">
                {errors.password.message as string}
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">
                {variant === "reset_password"
                  ? "Enter your new password"
                  : "Create a password for your account"}
              </p>
            )}
          </div>

          {claimError && (
            <div className="flex gap-2 items-start p-3 rounded-md bg-destructive/10 border border-destructive/30 text-destructive">
              <Info className="h-4 w-4 mt-0.5 shrink-0" />
              <div className="text-sm">{claimError}</div>
            </div>
          )}

          <div className="mt-6">
            <Button type="submit" disabled={isPending}>
              {isPending && (
                <LoaderCircle
                  className="h-4 w-4 animate-spin mr-2"
                  aria-label="loading"
                  role="img"
                />
              )}
              {variant === "reset_password" ? "Reset Password" : "Sign Up"}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}

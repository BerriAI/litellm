import React from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Info } from "lucide-react";
import { Form, Input } from "antd";

type OnboardingFormBodyProps = {
  variant: "signup" | "reset_password";
  userEmail: string;
  isPending: boolean;
  claimError: string | null;
  onSubmit: (values: { password: string }) => void;
};

export function OnboardingFormBody({
  variant,
  userEmail,
  isPending,
  claimError,
  onSubmit,
}: OnboardingFormBodyProps) {
  const [form] = Form.useForm();

  React.useEffect(() => {
    if (userEmail) form.setFieldValue("user_email", userEmail);
  }, [userEmail, form]);

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

        <Form
          className="mt-10 mb-5"
          layout="vertical"
          form={form}
          onFinish={(values) => onSubmit({ password: values.password })}
        >
          <Form.Item label="Email Address" name="user_email">
            <Input type="email" disabled />
          </Form.Item>

          <Form.Item
            label="Password"
            name="password"
            rules={[
              { required: true, message: "password required to sign up" },
            ]}
            help={
              variant === "reset_password"
                ? "Enter your new password"
                : "Create a password for your account"
            }
          >
            <Input.Password />
          </Form.Item>

          {claimError && (
            <div className="flex gap-2 items-start p-3 rounded-md bg-destructive/10 border border-destructive/30 text-destructive mb-4">
              <Info className="h-4 w-4 mt-0.5 shrink-0" />
              <div className="text-sm">{claimError}</div>
            </div>
          )}

          <div className="mt-10">
            <Button type="submit" disabled={isPending}>
              {isPending
                ? "…"
                : variant === "reset_password"
                  ? "Reset Password"
                  : "Sign Up"}
            </Button>
          </div>
        </Form>
      </Card>
    </div>
  );
}

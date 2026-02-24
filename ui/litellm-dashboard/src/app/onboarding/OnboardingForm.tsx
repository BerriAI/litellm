"use client";
import React from "react";
import { useSearchParams } from "next/navigation";
import { jwtDecode } from "jwt-decode";
import { Alert, Button, Card, Form, Input, Spin, Typography } from "antd";
import { useOnboardingCredentials, useClaimOnboardingToken } from "@/app/(dashboard)/hooks/onboarding/useOnboarding";
import { getProxyBaseUrl } from "@/components/networking";

type OnboardingFormProps = {
  variant: "signup" | "reset_password";
};

export function OnboardingForm({ variant }: OnboardingFormProps) {
  const [form] = Form.useForm();
  const searchParams = useSearchParams()!;
  const inviteId = searchParams.get("invitation_id");

  const {
    data: credentialsData,
    isLoading: isCredentialsLoading,
    isError: isCredentialsError,
  } = useOnboardingCredentials(inviteId);

  const { mutate: claimToken, isPending } = useClaimOnboardingToken();

  const decoded = credentialsData?.token
    ? (jwtDecode(credentialsData.token) as { [key: string]: any })
    : null;
  const userEmail = decoded?.user_email ?? "";
  const userId: string | null = decoded?.user_id ?? null;
  const accessToken: string | null = decoded?.key ?? null;
  const jwtToken: string | null = credentialsData?.token ?? null;

  React.useEffect(() => {
    if (userEmail) form.setFieldValue("user_email", userEmail);
  }, [userEmail, form]);

  const handleSubmit = (formValues: { password: string }) => {
    if (!accessToken || !jwtToken || !userId || !inviteId) return;

    claimToken(
      { accessToken, inviteId, userId, password: formValues.password },
      {
        onSuccess: () => {
          document.cookie = "token=" + jwtToken;
          const proxyBaseUrl = getProxyBaseUrl();
          window.location.href = proxyBaseUrl
            ? `${proxyBaseUrl}/ui/?login=success`
            : "/ui/?login=success";
        },
      }
    );
  };

  if (isCredentialsLoading) {
    return (
      <div className="mx-auto w-full max-w-md mt-10 flex justify-center">
        <Spin size="large" />
      </div>
    );
  }

  if (isCredentialsError) {
    return (
      <div className="mx-auto w-full max-w-md mt-10">
        <Alert
          type="error"
          message="Failed to load invitation"
          description="The invitation link may be invalid or expired."
          showIcon
        />
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card>
        <Typography.Title level={5} className="text-center mb-5">
          🚅 LiteLLM
        </Typography.Title>
        <Typography.Title level={3}>
          {variant === "reset_password" ? "Reset Password" : "Sign Up"}
        </Typography.Title>
        <Typography.Text>
          {variant === "reset_password"
            ? "Reset your password to access Admin UI."
            : "Claim your user account to login to Admin UI."}
        </Typography.Text>

        {variant === "signup" && (
          <Alert
            className="mt-4"
            type="info"
            message="SSO"
            description={
              <div className="flex justify-between items-center">
                <span>SSO is under the Enterprise Tier.</span>
                <Button
                  type="primary"
                  size="small"
                  href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                  target="_blank"
                >
                  Get Free Trial
                </Button>
              </div>
            }
            showIcon
          />
        )}

        <Form className="mt-10 mb-5" layout="vertical" form={form} onFinish={handleSubmit}>
          <Form.Item label="Email Address" name="user_email">
            <Input type="email" disabled />
          </Form.Item>

          <Form.Item
            label="Password"
            name="password"
            rules={[{ required: true, message: "password required to sign up" }]}
            help={
              variant === "reset_password"
                ? "Enter your new password"
                : "Create a password for your account"
            }
          >
            <Input.Password />
          </Form.Item>

          <div className="mt-10">
            <Button htmlType="submit" loading={isPending}>
              {variant === "reset_password" ? "Reset Password" : "Sign Up"}
            </Button>
          </div>
        </Form>
      </Card>
    </div>
  );
}

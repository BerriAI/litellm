import React from "react";
import { Alert, Button, Card, Form, Input, Typography } from "antd";
import { useTranslation } from "react-i18next";

type OnboardingFormBodyProps = {
  variant: "signup" | "reset_password";
  userEmail: string;
  isPending: boolean;
  claimError: string | null;
  onSubmit: (values: { password: string }) => void;
};

export function OnboardingFormBody({ variant, userEmail, isPending, claimError, onSubmit }: OnboardingFormBodyProps) {
  const { t } = useTranslation();
  const [form] = Form.useForm();

  React.useEffect(() => {
    if (userEmail) form.setFieldValue("user_email", userEmail);
  }, [userEmail, form]);

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card>
        <Typography.Title level={5} className="text-center mb-5">
          🚅 LiteLLM
        </Typography.Title>
        <Typography.Title level={3}>
          {variant === "reset_password"
            ? t("pages.onboardingFormBody.resetPasswordTitle")
            : t("pages.onboardingFormBody.signUpTitle")}
        </Typography.Title>
        <Typography.Text>
          {variant === "reset_password"
            ? t("pages.onboardingFormBody.resetPasswordSubtitle")
            : t("pages.onboardingFormBody.signUpSubtitle")}
        </Typography.Text>

        {variant === "signup" && (
          <Alert
            className="mt-4"
            type="info"
            message="SSO"
            description={
              <div className="flex justify-between items-center">
                <span>{t("pages.onboardingFormBody.ssoDescription")}</span>
                <Button type="primary" size="small" href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                  {t("pages.onboardingFormBody.getFreeTrial")}
                </Button>
              </div>
            }
            showIcon
          />
        )}

        <Form
          className="mt-10 mb-5"
          layout="vertical"
          form={form}
          onFinish={(values) => onSubmit({ password: values.password })}
        >
          <Form.Item label={t("pages.onboardingFormBody.emailAddressLabel")} name="user_email">
            <Input type="email" disabled />
          </Form.Item>

          <Form.Item
            label={t("pages.onboardingFormBody.passwordLabel")}
            name="password"
            rules={[{ required: true, message: t("pages.onboardingFormBody.passwordRequired") }]}
            help={
              variant === "reset_password"
                ? t("pages.onboardingFormBody.passwordHelpReset")
                : t("pages.onboardingFormBody.passwordHelpSignup")
            }
          >
            <Input.Password />
          </Form.Item>

          {claimError && <Alert type="error" message={claimError} showIcon className="mb-4" />}

          <div className="mt-10">
            <Button htmlType="submit" loading={isPending}>
              {variant === "reset_password"
                ? t("pages.onboardingFormBody.resetPasswordTitle")
                : t("pages.onboardingFormBody.signUpTitle")}
            </Button>
          </div>
        </Form>
      </Card>
    </div>
  );
}

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
  const { t } = useTranslation("auth");
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
          {variant === "reset_password" ? t("onboarding.resetTitle") : t("onboarding.signupTitle")}
        </Typography.Title>
        <Typography.Text>
          {variant === "reset_password" ? t("onboarding.resetDescription") : t("onboarding.signupDescription")}
        </Typography.Text>

        {variant === "signup" && (
          <Alert
            className="mt-4"
            type="info"
            message="SSO"
            description={
              <div className="flex justify-between items-center">
                <span>{t("onboarding.ssoTier")}</span>
                <Button type="primary" size="small" href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                  {t("onboarding.freeTrial")}
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
          <Form.Item label={t("onboarding.email")} name="user_email">
            <Input type="email" disabled />
          </Form.Item>

          <Form.Item
            label={t("onboarding.password")}
            name="password"
            rules={[{ required: true, message: t("onboarding.passwordRequired") }]}
            help={variant === "reset_password" ? t("onboarding.newPasswordHelp") : t("onboarding.createPasswordHelp")}
          >
            <Input.Password />
          </Form.Item>

          {claimError && <Alert type="error" message={claimError} showIcon className="mb-4" />}

          <div className="mt-10">
            <Button htmlType="submit" loading={isPending}>
              {variant === "reset_password" ? t("onboarding.resetSubmit") : t("onboarding.signupSubmit")}
            </Button>
          </div>
        </Form>
      </Card>
    </div>
  );
}

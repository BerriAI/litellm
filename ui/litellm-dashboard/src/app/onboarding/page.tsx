"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  Card,
  Title,
  Text,
  TextInput,
  Callout,
  Button,
  Grid,
  Col,
} from "@tremor/react";
import { RiAlarmWarningLine, RiCheckboxCircleLine } from "@remixicon/react";
import {
  invitationClaimCall,
  userUpdateUserCall,
  getOnboardingCredentials,
  claimOnboardingToken,
} from "@/components/networking";
import { jwtDecode } from "jwt-decode";
import { Form, Button as Button2, message } from "antd";
export default function Onboarding() {
  const [form] = Form.useForm();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const inviteID = searchParams.get("id");
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [defaultUserEmail, setDefaultUserEmail] = useState<string>("");
  const [userEmail, setUserEmail] = useState<string>("");
  const [userID, setUserID] = useState<string | null>(null);
  const [loginUrl, setLoginUrl] = useState<string>("");
  const [jwtToken, setJwtToken] = useState<string>("");

  useEffect(() => {
    if (!inviteID) {
      return;
    }
    getOnboardingCredentials(inviteID).then((data) => {
      const login_url = data.login_url;
      console.log("login_url:", login_url);
      setLoginUrl(login_url);

      const token = data.token;
      const decoded = jwtDecode(token) as { [key: string]: any };
      setJwtToken(token);

      console.log("decoded:", decoded);
      setAccessToken(decoded.key);

      console.log("decoded user email:", decoded.user_email);
      const user_email = decoded.user_email;
      setUserEmail(user_email);

      const user_id = decoded.user_id;
      setUserID(user_id);
    });
  }, [inviteID]);

  const handleSubmit = (formValues: Record<string, any>) => {
    console.log(
      "in handle submit. accessToken:",
      accessToken,
      "token:",
      jwtToken,
      "formValues:",
      formValues
    );
    if (!accessToken || !jwtToken) {
      return;
    }

    formValues.user_email = userEmail;

    if (!userID || !inviteID) {
      return;
    }
    claimOnboardingToken(
      accessToken,
      inviteID,
      userID,
      formValues.password
    ).then((data) => {
      let litellm_dashboard_ui = "/ui/";
      const user_id = data.data?.user_id || data.user_id;
      litellm_dashboard_ui += "?userID=" + user_id + "&token=" + jwtToken;
      console.log("redirecting to:", litellm_dashboard_ui);

      window.location.href = litellm_dashboard_ui;
    });

    // redirect to login page
  };
  return (
    <div className="mx-auto max-w-md mt-10">
      <Card>
        <Title className="text-sm mb-5 text-center">🚅 LiteLLM</Title>
        <Title className="text-xl">Sign up</Title>
        <Text>Claim your user account to login to Admin UI.</Text>

        <Callout
          className="mt-4"
          title="SSO"
          icon={RiCheckboxCircleLine}
          color="sky"
        >
          <Grid numItems={2} className="flex justify-between items-center">
            <Col>SSO is under the Enterprise Tirer.</Col>

            <Col>
              <Button variant="primary" className="mb-2">
                <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                  Get Free Trial
                </a>
              </Button>
            </Col>
          </Grid>
        </Callout>

        <Form
          className="mt-10 mb-5 mx-auto"
          layout="vertical"
          onFinish={handleSubmit}
        >
          <>
            <Form.Item label="Email Address" name="user_email">
              <TextInput
                type="email"
                disabled={true}
                value={userEmail}
                defaultValue={userEmail}
                className="max-w-md"
              />
            </Form.Item>

            <Form.Item
              label="Password"
              name="password"
              rules={[
                { required: true, message: "password required to sign up" },
              ]}
              help="Create a password for your account"
            >
              <TextInput placeholder="" type="password" className="max-w-md" />
            </Form.Item>
          </>

          <div className="mt-10">
            <Button2 htmlType="submit">Sign Up</Button2>
          </div>
        </Form>
      </Card>
    </div>
  );
}

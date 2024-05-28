"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Card, Title, Text, TextInput } from "@tremor/react";
import {
  invitationClaimCall,
  userUpdateUserCall,
} from "@/components/networking";
import { jwtDecode } from "jwt-decode";
import { Form, Button as Button2, message } from "antd";
export default function Onboarding() {
  const [form] = Form.useForm();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const user_email = searchParams.get("user_email");
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [defaultUserEmail, setDefaultUserEmail] = useState<string>("");
  const router = useRouter();

  useEffect(() => {
    if (!token) {
      return;
    }
    const decoded = jwtDecode(token) as { [key: string]: any };

    setAccessToken(decoded.key);
  }, [token]);

  useEffect(() => {
    if (!user_email) {
      return;
    }
    setDefaultUserEmail(user_email);
  }, [user_email]);

  const handleSubmit = (formValues: Record<string, any>) => {
    if (!accessToken || !token) {
      return;
    }

    userUpdateUserCall(accessToken, formValues, null).then((data) => {
      let litellm_dashboard_ui = "";
      const user_id = data.data?.user_id || data.user_id;

      litellm_dashboard_ui += "?userID=" + user_id + "&token=" + token;

      router.replace(litellm_dashboard_ui);
    });
  };
  return (
    <div className="mx-auto max-w-md mt-10">
      <Card>
        <Title className="text-sm mb-5 text-center">🚅 LiteLLM</Title>
        <Title className="text-xl">Sign up</Title>
        <Text>Claim your user account to login to Admin UI.</Text>
        <Form
          className="mt-10 mb-5 mx-auto"
          layout="vertical"
          onFinish={handleSubmit}
        >
          <>
            <Form.Item
              label="Email Address"
              name="user_email"
              rules={[{ required: true, message: "Set the user email" }]}
              help="required"
            >
              <TextInput
                placeholder={defaultUserEmail}
                type="email"
                className="max-w-md"
              />
            </Form.Item>

            <Form.Item
              label="Password"
              name="password"
              rules={[{ required: true, message: "Set the user password" }]}
              help="required"
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

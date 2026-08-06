import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Form } from "antd";
import PassthroughAuthorizeSection from "./PassthroughAuthorizeSection";

const WithForm: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [form] = Form.useForm();
  return <Form form={form}>{children}</Form>;
};

const noopFlow = { startOAuthFlow: () => {}, status: "idle", error: null, tokenResponse: null };

describe("PassthroughAuthorizeSection credential-class-aware copy", () => {
  it("shows keep-existing copy when the credential class is unchanged (true_passthrough <-> oauth_delegate)", () => {
    render(
      <WithForm>
        <PassthroughAuthorizeSection
          authType="oauth_delegate"
          oauthFlow={noopFlow}
          isEditing
          savedAuthType="true_passthrough"
        />
      </WithForm>,
    );
    expect(screen.getByPlaceholderText("Leave blank to keep the currently saved app (if any)")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Leave blank to keep the currently saved secret (if any)")).toBeInTheDocument();
  });

  it("shows the discard warning copy when switching from a different class (oauth2 -> true_passthrough)", () => {
    render(
      <WithForm>
        <PassthroughAuthorizeSection
          authType="true_passthrough"
          oauthFlow={noopFlow}
          isEditing
          savedAuthType="oauth2"
        />
      </WithForm>,
    );
    expect(screen.getByPlaceholderText("Leave blank to use dynamic client registration")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Leave blank for public clients / PKCE")).toBeInTheDocument();
    expect(screen.getByText(/Switching the auth type discards the previously saved app/)).toBeInTheDocument();
  });

  it("shows the keep+warn banner when the upstream may no longer match", () => {
    render(
      <WithForm>
        <PassthroughAuthorizeSection authType="true_passthrough" oauthFlow={noopFlow} appMayNotMatchUpstream />
      </WithForm>,
    );
    expect(screen.getByText(/registered for the previous upstream/)).toBeInTheDocument();
  });
});

import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { Form } from "antd";

import MCPPermissionManagement from "./MCPPermissionManagement";

const defaultProps = {
  availableAccessGroups: [],
  mcpServer: null,
  searchValue: "",
  setSearchValue: () => {},
  getAccessGroupOptions: () => [],
};

describe("MCPPermissionManagement", () => {
  const expandPanel = async () => {
    const user = userEvent.setup();
    const headerButton = screen.getByRole("button", {
      name: /permission management/i,
    });
    await user.click(headerButton);
    return user;
  };

  const renderWithForm = (props = {}) => {
    const Wrapper: React.FC = ({ children }) => {
      const [form] = Form.useForm();
      return (
        <Form form={form} initialValues={{ allow_all_keys: false }}>
          {children}
        </Form>
      );
    };

    return render(
      <Wrapper>
        <MCPPermissionManagement {...defaultProps} {...props} />
      </Wrapper>,
    );
  };

  it("should default allow_all_keys switch to unchecked for new servers", async () => {
    renderWithForm();
    await expandPanel();
    // Find the switch associated with "Allow All LiteLLM Keys" text
    // The first switch in the component is for allow_all_keys
    const switches = screen.getAllByRole("switch");
    const toggle = switches[0];
    expect(toggle).toHaveAttribute("aria-checked", "false");
  });

  const renderWithInitialValues = (initialValues: Record<string, unknown>, props = {}) => {
    const Wrapper: React.FC = ({ children }) => {
      const [form] = Form.useForm();
      return (
        <Form form={form} initialValues={initialValues}>
          {/* In the real app auth_type is registered by the parent form; the
              component only watches it. Register a hidden field here so
              Form.useWatch("auth_type") resolves the initial value. */}
          <Form.Item name="auth_type" hidden>
            <input />
          </Form.Item>
          {children}
        </Form>
      );
    };
    return render(
      <Wrapper>
        <MCPPermissionManagement {...defaultProps} {...props} />
      </Wrapper>,
    );
  };

  it("shows only the oauth2 PKCE-delegation toggle for oauth2 servers", async () => {
    renderWithInitialValues({ allow_all_keys: false, auth_type: "oauth2" });
    await expandPanel();
    expect(screen.getByText("Delegate auth to upstream (PKCE passthrough)")).toBeInTheDocument();
    // The non-oauth2 pass-through toggle must NOT appear for oauth2 servers.
    expect(screen.queryByText("OAuth pass-through")).not.toBeInTheDocument();
  });

  it("shows only the OAuth pass-through toggle for none-auth servers forwarding Authorization", async () => {
    renderWithInitialValues({
      allow_all_keys: false,
      auth_type: "none",
      extra_headers: ["Authorization"],
    });
    await expandPanel();
    expect(screen.getByText("OAuth pass-through")).toBeInTheDocument();
    // The oauth2-only PKCE delegation toggle must NOT appear here.
    expect(screen.queryByText("Delegate auth to upstream (PKCE passthrough)")).not.toBeInTheDocument();
  });

  it("hides both upstream-auth toggles for none-auth servers without an Authorization header", async () => {
    renderWithInitialValues({
      allow_all_keys: false,
      auth_type: "none",
      extra_headers: ["x-api-key"],
    });
    await expandPanel();
    expect(screen.queryByText("OAuth pass-through")).not.toBeInTheDocument();
    expect(screen.queryByText("Delegate auth to upstream (PKCE passthrough)")).not.toBeInTheDocument();
  });

  it("should reflect allow_all_keys when editing an existing server", async () => {
    renderWithForm({
      mcpServer: {
        server_id: "server-1",
        url: "https://example.com",
        created_at: "2024-01-01T00:00:00Z",
        created_by: "user",
        updated_at: "2024-01-01T00:00:00Z",
        updated_by: "user",
        allow_all_keys: true,
      },
    });

    const user = await expandPanel();
    // Find the switch associated with "Allow All LiteLLM Keys" text
    // The first switch in the component is for allow_all_keys
    const switches = screen.getAllByRole("switch");
    const toggle = switches[0];
    expect(toggle).toHaveAttribute("aria-checked", "true");

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-checked", "false");
  });
});

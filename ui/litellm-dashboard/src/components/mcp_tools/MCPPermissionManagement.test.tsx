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

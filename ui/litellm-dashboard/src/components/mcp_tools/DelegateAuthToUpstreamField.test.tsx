import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { Form, Input } from "antd";

import DelegateAuthToUpstreamField from "./DelegateAuthToUpstreamField";

const PASSTHROUGH_NOTICE = /reachable without a LiteLLM login/i;
const INTERNAL_WARNING = /Internal server with upstream OAuth delegation/i;

const Harness: React.FC<{ initialValues?: Record<string, any> }> = ({ initialValues }) => {
  const [form] = Form.useForm();
  return (
    <Form form={form} initialValues={initialValues}>
      <Form.Item name="available_on_public_internet" hidden>
        <Input />
      </Form.Item>
      <DelegateAuthToUpstreamField />
    </Form>
  );
};

const renderField = (initialValues: Record<string, any> = {}) => render(<Harness initialValues={initialValues} />);

describe("DelegateAuthToUpstreamField", () => {
  it("renders the toggle off with no passthrough notice by default", () => {
    renderField();
    expect(screen.getByText("Delegate auth to upstream (PKCE passthrough)")).toBeInTheDocument();
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
    expect(screen.queryByText(PASSTHROUGH_NOTICE)).not.toBeInTheDocument();
  });

  it("shows the passthrough notice once delegation is enabled", async () => {
    const user = userEvent.setup();
    renderField({ available_on_public_internet: true });

    expect(screen.queryByText(PASSTHROUGH_NOTICE)).not.toBeInTheDocument();
    await user.click(screen.getByRole("switch"));

    expect(screen.getByText(PASSTHROUGH_NOTICE)).toBeInTheDocument();
    expect(screen.queryByText(INTERNAL_WARNING)).not.toBeInTheDocument();
  });

  it("warns when delegation is on for an internal-only server", async () => {
    const user = userEvent.setup();
    renderField({ available_on_public_internet: false });
    await user.click(screen.getByRole("switch"));

    expect(screen.getByText(PASSTHROUGH_NOTICE)).toBeInTheDocument();
    expect(screen.getByText(INTERNAL_WARNING)).toBeInTheDocument();
  });

  it("reflects a saved delegate_auth_to_upstream value when editing", () => {
    renderField({ delegate_auth_to_upstream: true, available_on_public_internet: true });
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText(PASSTHROUGH_NOTICE)).toBeInTheDocument();
  });
});

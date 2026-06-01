import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Form, Switch } from "antd";

import DelegateAuthUpstreamField from "./DelegateAuthUpstreamField";

const WARNING = /Internal server with upstream OAuth delegation/i;

// The warning watches `available_on_public_internet`, a field owned by the host
// form. Register it (hidden) so Form.useWatch resolves it from initialValues,
// mirroring how the real create/edit forms provide it.
const renderField = (initialValues: Record<string, any> = {}) => {
  const Wrapper: React.FC = () => {
    const [form] = Form.useForm();
    return (
      <Form form={form} initialValues={initialValues}>
        <Form.Item name="available_on_public_internet" valuePropName="checked" hidden>
          <Switch />
        </Form.Item>
        <DelegateAuthUpstreamField initialValue={initialValues.delegate_auth_to_upstream ?? false} />
      </Form>
    );
  };
  return render(<Wrapper />);
};

describe("DelegateAuthUpstreamField", () => {
  it("renders the toggle, unchecked by default", () => {
    renderField();
    expect(screen.getByText("Delegate auth to upstream (PKCE passthrough)")).toBeInTheDocument();
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
  });

  it("reflects an initialValue of true", () => {
    renderField({ delegate_auth_to_upstream: true });
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
  });

  it("hides the internal-only warning when delegation is off", () => {
    renderField({ delegate_auth_to_upstream: false, available_on_public_internet: false });
    expect(screen.queryByText(WARNING)).not.toBeInTheDocument();
  });

  it("hides the internal-only warning when the server is public", () => {
    renderField({ delegate_auth_to_upstream: true, available_on_public_internet: true });
    expect(screen.queryByText(WARNING)).not.toBeInTheDocument();
  });

  it("shows the internal-only warning only when delegating on an internal-only server", () => {
    renderField({ delegate_auth_to_upstream: true, available_on_public_internet: false });
    expect(screen.getByText(WARNING)).toBeInTheDocument();
  });
});

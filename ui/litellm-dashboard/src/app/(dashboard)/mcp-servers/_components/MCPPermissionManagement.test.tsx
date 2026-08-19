import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { FormProvider, useForm } from "react-hook-form";

import {
  MountedFormField,
  MountedFormProvider,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import MCPPermissionManagement from "./MCPPermissionManagement";

const Wrapper: React.FC<{ children: React.ReactNode; defaultValues: MountedFormValues; withAuthType?: boolean }> = ({
  children,
  defaultValues,
  withAuthType = false,
}) => {
  const form = useForm<MountedFormValues>({ defaultValues });
  const registry = useMountRegistry();
  return (
    <FormProvider {...form}>
      <MountedFormProvider value={{ control: form.control, registry }}>
        {withAuthType && (
          <MountedFormField name="auth_type" bare>
            {(field) => <input type="hidden" value={String(field.value ?? "")} onChange={field.onChange} />}
          </MountedFormField>
        )}
        {children}
      </MountedFormProvider>
    </FormProvider>
  );
};

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

  const renderWithForm = (props = {}) =>
    render(
      <Wrapper defaultValues={{ allow_all_keys: false }}>
        <MCPPermissionManagement {...defaultProps} {...props} />
      </Wrapper>,
    );

  it("should default allow_all_keys switch to unchecked for new servers", async () => {
    renderWithForm();
    await expandPanel();
    // Find the switch associated with "Allow All LiteLLM Keys" text
    // The first switch in the component is for allow_all_keys
    const switches = screen.getAllByRole("switch");
    const toggle = switches[0];
    expect(toggle).not.toBeChecked();
  });

  const renderWithInitialValues = (initialValues: Record<string, unknown>, props = {}) =>
    render(
      <Wrapper defaultValues={initialValues} withAuthType>
        <MCPPermissionManagement {...defaultProps} {...props} />
      </Wrapper>,
    );

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
    expect(toggle).toBeChecked();

    await user.click(toggle);
    expect(toggle).not.toBeChecked();
  });
});

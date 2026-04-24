import React from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FormProvider, useForm } from "react-hook-form";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../../tests/test-utils";
import BaseSSOSettingsForm, {
  renderProviderFields,
  SSOSettingsFormValues,
} from "./BaseSSOSettingsForm";

const TestWrapper: React.FC<{
  defaultValues?: Partial<SSOSettingsFormValues>;
  onSubmit?: (values: SSOSettingsFormValues) => void;
}> = ({ defaultValues, onSubmit }) => {
  const form = useForm<SSOSettingsFormValues>({
    defaultValues: { ...defaultValues },
    mode: "onSubmit",
  });
  const handleSubmit = form.handleSubmit((values) => {
    onSubmit?.(values);
  });
  return (
    <FormProvider {...form}>
      <form onSubmit={handleSubmit}>
        <BaseSSOSettingsForm />
        <button type="submit">submit</button>
      </form>
    </FormProvider>
  );
};

const selectProvider = async (providerText: RegExp) => {
  const user = userEvent.setup();
  const trigger = screen.getByRole("combobox", { name: /sso provider/i });
  await user.click(trigger);
  const option = await screen.findByRole("option", { name: providerText });
  await user.click(option);
};

describe("BaseSSOSettingsForm", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render top-level fields", () => {
    renderWithProviders(<TestWrapper />);

    expect(screen.getByText("SSO Provider")).toBeInTheDocument();
    expect(screen.getByText("Proxy Admin Email")).toBeInTheDocument();
    expect(screen.getByText("Proxy Base URL")).toBeInTheDocument();
  });

  it("should render provider fields when provider is selected", async () => {
    renderWithProviders(<TestWrapper />);

    await selectProvider(/google sso/i);

    await waitFor(() => {
      expect(screen.getByText("Google Client ID")).toBeInTheDocument();
      expect(screen.getByText("Google Client Secret")).toBeInTheDocument();
    });
  });

  it("should show role mappings checkbox for okta provider", async () => {
    renderWithProviders(<TestWrapper />);

    await selectProvider(/okta/i);

    await waitFor(() => {
      expect(screen.getByText("Use Role Mappings")).toBeInTheDocument();
    });
  });

  it("should validate proxy base url format", async () => {
    renderWithProviders(<TestWrapper />);

    const urlInput = screen.getByPlaceholderText("https://example.com");
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "invalid-url" } });
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /submit/i }));
    });

    await waitFor(() => {
      expect(
        screen.getByText(/URL must start with http:\/\/ or https:\/\//i),
      ).toBeInTheDocument();
    });
  });

  it("should validate proxy base url trailing slash", async () => {
    renderWithProviders(<TestWrapper />);

    const urlInput = screen.getByPlaceholderText("https://example.com");
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://example.com/" } });
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /submit/i }));
    });

    await waitFor(() => {
      expect(
        screen.getByText(/URL must not end with a trailing slash/i),
      ).toBeInTheDocument();
    });
  });

  it("should show role mappings fields when use_role_mappings is checked for generic provider", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TestWrapper />);

    await selectProvider(/generic sso/i);

    await waitFor(() => {
      expect(screen.getByText("Use Role Mappings")).toBeInTheDocument();
    });

    const checkbox = screen.getByLabelText("Use Role Mappings");
    await user.click(checkbox);

    await waitFor(() => {
      expect(screen.getByText("Group Claim")).toBeInTheDocument();
      expect(screen.getByText("Default Role")).toBeInTheDocument();
    });
  });

  it("should show team mappings checkbox for okta provider", async () => {
    renderWithProviders(<TestWrapper />);

    await selectProvider(/okta/i);

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });
  });

  it("should show team mappings checkbox for generic provider", async () => {
    renderWithProviders(<TestWrapper />);

    await selectProvider(/generic sso/i);

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });
  });

  it("should show team IDs JWT field when use_team_mappings is checked for okta provider", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TestWrapper />);

    await selectProvider(/okta/i);

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });

    const checkbox = screen.getByLabelText("Use Team Mappings");
    await user.click(checkbox);

    await waitFor(() => {
      expect(screen.getByText("Team IDs JWT Field")).toBeInTheDocument();
    });
  });

  it("should not show team mappings checkbox for google provider", async () => {
    renderWithProviders(<TestWrapper />);

    await selectProvider(/google sso/i);

    await waitFor(() => {
      expect(screen.getByText("Google Client ID")).toBeInTheDocument();
    });

    expect(screen.queryByText("Use Team Mappings")).not.toBeInTheDocument();
  });
});

describe("renderProviderFields", () => {
  it("should return null for unknown provider", () => {
    const result = renderProviderFields("unknown");
    expect(result).toBeNull();
  });

  it("should return fields for google provider", () => {
    const result = renderProviderFields("google");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(2);
  });

  it("should return fields for microsoft provider", () => {
    const result = renderProviderFields("microsoft");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(3);
  });

  it("should return fields for okta provider", () => {
    const result = renderProviderFields("okta");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(5);
  });

  it("should return fields for generic provider", () => {
    const result = renderProviderFields("generic");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(5);
  });
});

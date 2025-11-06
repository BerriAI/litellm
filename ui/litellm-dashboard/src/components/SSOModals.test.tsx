import { render, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, beforeAll } from "vitest";
import { Form } from "antd";
import SSOModals from "./SSOModals";
import React from "react";

// Mock window.matchMedia for Ant Design components
beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {}, // deprecated
      removeListener: () => {}, // deprecated
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => true,
    }),
  });
});

describe("SSOModals", () => {
  it("should render the SSOModals component", () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();

      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    const { getByText } = render(<TestWrapper />);
    expect(getByText("Add SSO")).toBeInTheDocument();
  });

  it("should have a validation error if the proxy base url is not a valid URL", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    const { getByLabelText, getByText, container } = render(<TestWrapper />);

    // Find and interact with the SSO provider select
    const ssoProviderSelect = container.querySelector("#sso_provider");
    if (ssoProviderSelect) {
      fireEvent.mouseDown(ssoProviderSelect);
      // Wait for dropdown and select Google
      await waitFor(() => {
        const googleOption = getByText("Google SSO");
        fireEvent.click(googleOption);
      });
    }

    // Fill in the email field
    const emailInput = getByLabelText("Proxy Admin Email");
    fireEvent.change(emailInput, { target: { value: "test@example.com" } });

    // Fill in an invalid URL
    const urlInput = getByLabelText("PROXY BASE URL");
    fireEvent.change(urlInput, { target: { value: "invalid-url" } });

    // Submit the form
    const saveButton = getByText("Save");
    fireEvent.click(saveButton);

    // Check for validation error
    await waitFor(() => {
      expect(getByText("URL must start with http:// or https://")).toBeInTheDocument();
    });
  });

  it("should automatically remove trailing slash from the proxy base url", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    const { getByLabelText, container } = render(<TestWrapper />);

    // Fill in the proxy base url with a trailing slash
    const urlInput = getByLabelText("PROXY BASE URL") as HTMLInputElement;
    fireEvent.change(urlInput, { target: { value: "https://example.com/" } });

    // Trigger blur to ensure normalization is applied
    fireEvent.blur(urlInput);

    // Check that the trailing slash was removed by the normalize function
    await waitFor(() => {
      expect(urlInput.value).toBe("https://example.com");
    });
  });
});

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
    const urlInput = getByLabelText("Proxy Base URL");
    fireEvent.change(urlInput, { target: { value: "invalid-url" } });

    // Submit the form
    const saveButton = getByText("Save");
    fireEvent.click(saveButton);

    // Check for validation error
    await waitFor(
      () => {
        expect(getByText("URL must start with http:// or https://")).toBeInTheDocument();
      },
      // The validation is based on a Promise, so we need to wait for it to resolve
      { timeout: 5000 },
    );
  });

  it("should show validation error if the proxy base url ends with a trailing slash", async () => {
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

    const { getByLabelText, getByText, findByText, container } = render(<TestWrapper />);

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

    // Fill in a URL with trailing slash
    const urlInput = getByLabelText("Proxy Base URL") as HTMLInputElement;
    fireEvent.change(urlInput, { target: { value: "https://example.com/" } });

    // Submit the form
    const saveButton = getByText("Save");
    fireEvent.click(saveButton);

    // Check for validation error using findByText for async rendering
    const errorMessage = await findByText("URL must not end with a trailing slash", {}, { timeout: 5000 });
    expect(errorMessage).toBeInTheDocument();
  });

  it("should allow typing https:// without interfering with slashes", async () => {
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

    const { getByLabelText } = render(<TestWrapper />);

    const urlInput = getByLabelText("Proxy Base URL") as HTMLInputElement;

    // Simulate user typing "https://"
    fireEvent.change(urlInput, { target: { value: "h" } });
    expect(urlInput.value).toBe("h");

    fireEvent.change(urlInput, { target: { value: "ht" } });
    expect(urlInput.value).toBe("ht");

    fireEvent.change(urlInput, { target: { value: "http" } });
    expect(urlInput.value).toBe("http");

    fireEvent.change(urlInput, { target: { value: "https" } });
    expect(urlInput.value).toBe("https");

    fireEvent.change(urlInput, { target: { value: "https:" } });
    expect(urlInput.value).toBe("https:");

    fireEvent.change(urlInput, { target: { value: "https:/" } });
    expect(urlInput.value).toBe("https:/");

    fireEvent.change(urlInput, { target: { value: "https://" } });
    expect(urlInput.value).toBe("https://");

    // Continue typing the domain
    fireEvent.change(urlInput, { target: { value: "https://example.com" } });
    expect(urlInput.value).toBe("https://example.com");
  });

  it("should only show URL format error for incomplete URLs, not trailing slash error", async () => {
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

    const { getByLabelText, getByText, queryByText, container, findByText } = render(<TestWrapper />);

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

    // Fill in an incomplete URL like "http:"
    const urlInput = getByLabelText("Proxy Base URL");
    fireEvent.change(urlInput, { target: { value: "http:" } });

    // Submit the form
    const saveButton = getByText("Save");
    fireEvent.click(saveButton);

    // Check that only the URL format error appears (use findByText for async rendering)
    const errorMessage = await findByText("URL must start with http:// or https://", {}, { timeout: 3000 });
    expect(errorMessage).toBeInTheDocument();

    // Verify the trailing slash error does NOT appear
    expect(queryByText("URL must not end with a trailing slash")).not.toBeInTheDocument();
  });
});

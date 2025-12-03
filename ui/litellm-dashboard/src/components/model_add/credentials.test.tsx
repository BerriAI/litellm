import { CredentialItem } from "@/components/networking";
import { render, screen, waitFor } from "@testing-library/react";
import { UploadProps } from "antd/es/upload";
import { describe, expect, it, vi } from "vitest";
import CredentialsPanel from "./credentials";

const DEFAULT_UPLOAD_PROPS = {} as UploadProps;

describe("CredentialsPanel", () => {
  it("should render", () => {
    const fetchCredentials = vi.fn(() => Promise.resolve());

    render(
      <CredentialsPanel
        accessToken="test-token"
        uploadProps={DEFAULT_UPLOAD_PROPS}
        credentialList={[]}
        fetchCredentials={fetchCredentials}
      />,
    );

    expect(screen.getByRole("button", { name: /add credential/i })).toBeInTheDocument();
  });

  it("should call fetchCredentials when accessToken exists", async () => {
    const fetchCredentials = vi.fn(() => Promise.resolve());

    render(
      <CredentialsPanel
        accessToken="test-token"
        uploadProps={DEFAULT_UPLOAD_PROPS}
        credentialList={[]}
        fetchCredentials={fetchCredentials}
      />,
    );

    await waitFor(() => {
      expect(fetchCredentials).toHaveBeenCalledWith("test-token");
    });
  });

  it("should display provided credentials", () => {
    const fetchCredentials = vi.fn(() => Promise.resolve());
    const credentials: CredentialItem[] = [
      {
        credential_name: "openai-key",
        credential_values: {},
        credential_info: { custom_llm_provider: "openai" },
      },
    ];

    render(
      <CredentialsPanel
        accessToken="another-token"
        uploadProps={DEFAULT_UPLOAD_PROPS}
        credentialList={credentials}
        fetchCredentials={fetchCredentials}
      />,
    );

    expect(screen.getByText("openai-key")).toBeInTheDocument();
  });

  it("should display empty state when no credentials are provided", () => {
    const fetchCredentials = vi.fn(() => Promise.resolve());

    render(
      <CredentialsPanel
        accessToken="test-token"
        uploadProps={DEFAULT_UPLOAD_PROPS}
        credentialList={[]}
        fetchCredentials={fetchCredentials}
      />,
    );

    expect(screen.getByText("No credentials configured")).toBeInTheDocument();
  });
});

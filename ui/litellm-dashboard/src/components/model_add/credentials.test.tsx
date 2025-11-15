import { CredentialItem } from "@/components/networking";
import { render, waitFor } from "@testing-library/react";
import { UploadProps } from "antd/es/upload";
import { describe, expect, it, vi } from "vitest";
import CredentialsPanel from "./credentials";

const DEFAULT_UPLOAD_PROPS = {} as UploadProps;

describe("CredentialsPanel", () => {
  it("renders without crashing and fetches credentials when token exists", async () => {
    const fetchCredentials = vi.fn(() => Promise.resolve());

    const { getByRole, getByText } = render(
      <CredentialsPanel
        accessToken="test-token"
        uploadProps={DEFAULT_UPLOAD_PROPS}
        credentialList={[]}
        fetchCredentials={fetchCredentials}
      />,
    );

    await waitFor(() => {
      expect(getByRole("button", { name: /add credential/i })).toBeInTheDocument();
      expect(getByText("Credential Name")).toBeInTheDocument();
      expect(getByText("Provider")).toBeInTheDocument();
    });
  });

  it("displays provided credentials and still calls the fetch helper", async () => {
    const fetchCredentials = vi.fn(() => Promise.resolve());
    const credentials: CredentialItem[] = [
      {
        credential_name: "openai-key",
        credential_values: {},
        credential_info: { custom_llm_provider: "openai" },
      },
    ];

    const { getByText } = render(
      <CredentialsPanel
        accessToken="another-token"
        uploadProps={DEFAULT_UPLOAD_PROPS}
        credentialList={credentials}
        fetchCredentials={fetchCredentials}
      />,
    );

    await waitFor(() => expect(getByText("openai-key")).toBeInTheDocument());
  });
});

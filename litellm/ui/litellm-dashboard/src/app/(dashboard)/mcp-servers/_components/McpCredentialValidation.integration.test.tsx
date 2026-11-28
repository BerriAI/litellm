import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import { MountedFormProvider, useMountRegistry } from "@/components/common_components/MountedFormField";
import { useZodForm } from "@/lib/forms/useZodForm";

import AwsSigV4Fields from "./AwsSigV4Fields";
import IdJagFormFields from "./IdJagFormFields";
import { McpFormProvider } from "./McpFormField";
import { useMcpFormSchema } from "./schema";

function CredentialForm({ children }: { children: ReactNode }) {
  const { schema, registerValidation } = useMcpFormSchema();
  const form = useZodForm(schema, { mode: "onChange" });
  const registry = useMountRegistry();

  return (
    <McpFormProvider form={form} registerValidation={registerValidation}>
      <MountedFormProvider value={{ control: form.control, registry }}>{children}</MountedFormProvider>
    </McpFormProvider>
  );
}

describe("MCP credential validation", () => {
  it("shows the missing AWS secret when an access key ID is entered", async () => {
    render(
      <CredentialForm>
        <AwsSigV4Fields />
      </CredentialForm>,
    );

    fireEvent.change(screen.getByLabelText("AWS Access Key ID"), { target: { value: "example-access-key-id" } });

    expect(await screen.findByText("Secret Access Key is required when Access Key ID is provided")).toBeVisible();
  });

  it("clears the ID-JAG secret error when the secret is rechecked with a private key set", async () => {
    render(
      <CredentialForm>
        <IdJagFormFields />
      </CredentialForm>,
    );
    const secret = screen.getByLabelText("Client Secret");
    const message = "Provide either a client secret or a client private key";

    fireEvent.change(secret, { target: { value: "example-secret" } });
    fireEvent.change(secret, { target: { value: "" } });
    expect(await screen.findByText(message)).toBeVisible();

    const privateKey = screen.getByLabelText("Client Private Key (PEM)");
    await act(async () => {
      fireEvent.change(privateKey, { target: { value: "example-private-key" } });
    });
    expect(screen.getByText(message)).toBeVisible();

    fireEvent.change(secret, { target: { value: "example-secret" } });
    fireEvent.change(secret, { target: { value: "" } });

    await waitFor(() => expect(screen.queryByText(message)).not.toBeInTheDocument());
  });
});

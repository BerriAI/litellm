import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CredentialItem } from "@/components/networking";

import CredentialsTable from "./CredentialsTable";

vi.mock("@/components/provider_info_helpers", () => ({
  getProviderLogoAndName: (provider: string) => {
    const providerMap: Record<string, { displayName: string; logo: string }> = {
      openai: { displayName: "OpenAI", logo: "/openai-logo.png" },
      azure: { displayName: "Azure", logo: "/azure-logo.png" },
    };
    return providerMap[provider] || { displayName: provider, logo: "" };
  },
}));

const mockCredentials: CredentialItem[] = [
  {
    credential_name: "b-openai-key",
    credential_values: {},
    credential_info: { custom_llm_provider: "openai" },
  },
  {
    credential_name: "a-azure-key",
    credential_values: {},
    credential_info: { custom_llm_provider: "azure" },
  },
];

const mockOnEdit = vi.fn();
const mockOnDelete = vi.fn();

const defaultProps = {
  credentials: mockCredentials,
  canModifyCredentials: true,
  onEdit: mockOnEdit,
  onDelete: mockOnDelete,
};

describe("CredentialsTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the data column headers", () => {
    render(<CredentialsTable {...defaultProps} />);
    for (const header of ["Credential Name", "Provider"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should display each credential name", () => {
    render(<CredentialsTable {...defaultProps} />);
    expect(screen.getByText("b-openai-key")).toBeInTheDocument();
    expect(screen.getByText("a-azure-key")).toBeInTheDocument();
  });

  it("should render provider display names from the logo helper", () => {
    render(<CredentialsTable {...defaultProps} />);
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();
  });

  it("should render a dash when a credential has no provider", () => {
    const credentials: CredentialItem[] = [
      { credential_name: "no-provider", credential_values: {}, credential_info: {} },
    ];
    render(<CredentialsTable {...defaultProps} credentials={credentials} />);
    const row = screen.getAllByRole("row").slice(1)[0];
    expect(within(row).getByText("-")).toBeInTheDocument();
  });

  it("should sort by credential name ascending by default", () => {
    render(<CredentialsTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("a-azure-key")).toBeInTheDocument();
    expect(within(rows[1]).getByText("b-openai-key")).toBeInTheDocument();
  });

  it("should display the empty state when there are no credentials", () => {
    render(<CredentialsTable {...defaultProps} credentials={[]} />);
    expect(screen.getByText("No credentials configured")).toBeInTheDocument();
  });

  it("should edit a credential through the actions menu", async () => {
    const user = userEvent.setup();
    render(<CredentialsTable {...defaultProps} />);
    await user.click(screen.getByTestId("credential-actions-b-openai-key"));
    await user.click(await screen.findByTestId("credential-action-edit"));
    expect(mockOnEdit).toHaveBeenCalledWith(mockCredentials[0]);
  });

  it("should delete a credential through the actions menu", async () => {
    const user = userEvent.setup();
    render(<CredentialsTable {...defaultProps} />);
    await user.click(screen.getByTestId("credential-actions-b-openai-key"));
    await user.click(await screen.findByTestId("credential-action-delete"));
    expect(mockOnDelete).toHaveBeenCalledWith(mockCredentials[0]);
  });

  it("should copy the credential name through the actions menu", async () => {
    const user = userEvent.setup();
    render(<CredentialsTable {...defaultProps} />);
    await user.click(screen.getByTestId("credential-actions-b-openai-key"));
    await user.click(await screen.findByTestId("credential-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("b-openai-key");
  });

  it("should not render the actions menu when the user cannot modify credentials", () => {
    render(<CredentialsTable {...defaultProps} canModifyCredentials={false} />);
    // Read parity: names still render...
    expect(screen.getByText("b-openai-key")).toBeInTheDocument();
    // ...but there is no per-row actions trigger.
    expect(screen.queryByTestId("credential-actions-b-openai-key")).not.toBeInTheDocument();
    expect(screen.queryByTestId("credential-actions-a-azure-key")).not.toBeInTheDocument();
  });
});

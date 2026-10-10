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

  it("should mark only the credentials the proxy federates as federated", () => {
    const credentials: CredentialItem[] = [
      {
        credential_name: "a-anthropic-federated",
        credential_values: { anthropic_federation_rule_id: "fdrl_stored", anthropic_organization_id: "org-stored" },
        credential_info: { custom_llm_provider: "anthropic" },
      },
      {
        credential_name: "b-anthropic-key",
        credential_values: { api_key: "sk-a****" },
        credential_info: { custom_llm_provider: "anthropic" },
      },
      {
        credential_name: "c-openai-federated",
        credential_values: { openai_service_account_id: "svc_stored" },
        credential_info: { custom_llm_provider: "openai" },
      },
      {
        credential_name: "d-openai-key-and-federation",
        credential_values: { api_key: "sk-p****", openai_service_account_id: "svc_stored" },
        credential_info: { custom_llm_provider: "openai" },
      },
    ];
    render(<CredentialsTable {...defaultProps} credentials={credentials} />);
    const rows = screen.getAllByRole("row").slice(1);
    const federatedNames = rows
      .filter((row) => within(row).queryByText("Workload identity federation") !== null)
      .map((row) => within(row).getAllByRole("cell")[0].textContent);
    expect(federatedNames).toEqual(["a-anthropic-federated", "c-openai-federated"]);
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

  it("should show the display name with the credential name beneath it, and the bare name when unset", () => {
    const credentials: CredentialItem[] = [
      {
        credential_name: "openai-prod",
        display_name: "Prod OpenAI",
        credential_values: {},
        credential_info: { custom_llm_provider: "openai" },
      },
      { credential_name: "plain-key", credential_values: {}, credential_info: { custom_llm_provider: "openai" } },
    ];
    render(<CredentialsTable {...defaultProps} credentials={credentials} />);

    const labeledRow = screen.getByRole("row", { name: /Prod OpenAI/ });
    expect(within(labeledRow).getByText("openai-prod")).toBeInTheDocument();
    const plainRow = screen.getByRole("row", { name: /plain-key/ });
    expect(within(plainRow).getAllByText("plain-key")).toHaveLength(1);
  });

  it("should sort by the display name when one is set", () => {
    const credentials: CredentialItem[] = [
      { credential_name: "a-key", display_name: "zulu", credential_values: {}, credential_info: {} },
      { credential_name: "b-key", credential_values: {}, credential_info: {} },
    ];
    render(<CredentialsTable {...defaultProps} credentials={credentials} />);

    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("b-key")).toBeInTheDocument();
    expect(within(rows[1]).getByText("zulu")).toBeInTheDocument();
  });

  it("should badge a config credential and block editing and deleting it", async () => {
    const user = userEvent.setup();
    const credentials: CredentialItem[] = [
      { credential_name: "from-config", source: "config", credential_values: {}, credential_info: {} },
      { credential_name: "from-db", source: "db", credential_values: {}, credential_info: {} },
    ];
    render(<CredentialsTable {...defaultProps} credentials={credentials} />);

    expect(within(screen.getByRole("row", { name: /from-config/ })).getByText("Config")).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /from-db/ })).queryByText("Config")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("credential-actions-from-config"));
    expect(await screen.findByTestId("credential-config-owned-hint")).toBeInTheDocument();
    const edit = screen.getByTestId("credential-action-edit");
    const remove = screen.getByTestId("credential-action-delete");
    expect(edit).toHaveAttribute("data-disabled");
    expect(remove).toHaveAttribute("data-disabled");
    await user.click(edit);
    await user.click(remove);
    expect(mockOnEdit).not.toHaveBeenCalled();
    expect(mockOnDelete).not.toHaveBeenCalled();
  });

  it("should keep editing enabled for a DB credential", async () => {
    const user = userEvent.setup();
    const credentials: CredentialItem[] = [
      { credential_name: "from-db", source: "db", credential_values: {}, credential_info: {} },
    ];
    render(<CredentialsTable {...defaultProps} credentials={credentials} />);

    await user.click(screen.getByTestId("credential-actions-from-db"));
    expect(await screen.findByTestId("credential-action-edit")).not.toHaveAttribute("data-disabled");
    expect(screen.queryByTestId("credential-config-owned-hint")).not.toBeInTheDocument();
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

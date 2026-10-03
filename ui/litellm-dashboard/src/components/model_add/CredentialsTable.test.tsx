import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CredentialItem } from "@/components/networking";

import { renderWithProviders, screen, waitFor, within } from "../../../tests/test-utils";

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
    renderWithProviders(<CredentialsTable {...defaultProps} />);
    for (const header of ["Credential Name", "Provider"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should display each credential name", () => {
    renderWithProviders(<CredentialsTable {...defaultProps} />);
    expect(screen.getByText("b-openai-key")).toBeInTheDocument();
    expect(screen.getByText("a-azure-key")).toBeInTheDocument();
  });

  it("should render provider display names from the logo helper", () => {
    renderWithProviders(<CredentialsTable {...defaultProps} />);
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();
  });

  it("should render a dash when a credential has no provider", () => {
    const credentials: CredentialItem[] = [
      { credential_name: "no-provider", credential_values: {}, credential_info: {} },
    ];
    renderWithProviders(<CredentialsTable {...defaultProps} credentials={credentials} />);
    const row = screen.getAllByRole("row").slice(1)[0];
    expect(within(row).getByText("-")).toBeInTheDocument();
  });

  it("should sort by credential name ascending by default", () => {
    renderWithProviders(<CredentialsTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("a-azure-key")).toBeInTheDocument();
    expect(within(rows[1]).getByText("b-openai-key")).toBeInTheDocument();
  });

  it("should display the empty state when there are no credentials", () => {
    renderWithProviders(<CredentialsTable {...defaultProps} credentials={[]} />);
    expect(screen.getByText("No credentials configured")).toBeInTheDocument();
  });

  it("should edit a credential through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CredentialsTable {...defaultProps} />);
    await user.click(screen.getByTestId("credential-actions-b-openai-key"));
    await user.click(await screen.findByTestId("credential-action-edit"));
    expect(mockOnEdit).toHaveBeenCalledWith(mockCredentials[0]);
  });

  it("should delete a credential through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CredentialsTable {...defaultProps} />);
    await user.click(screen.getByTestId("credential-actions-b-openai-key"));
    await user.click(await screen.findByTestId("credential-action-delete"));
    expect(mockOnDelete).toHaveBeenCalledWith(mockCredentials[0]);
  });

  it("should copy the credential name through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CredentialsTable {...defaultProps} />);
    await user.click(screen.getByTestId("credential-actions-b-openai-key"));
    await user.click(await screen.findByTestId("credential-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("b-openai-key");
  });

  it("should not render the actions menu when the user cannot modify credentials", () => {
    renderWithProviders(<CredentialsTable {...defaultProps} canModifyCredentials={false} />);
    // Read parity: names still render...
    expect(screen.getByText("b-openai-key")).toBeInTheDocument();
    // ...but there is no per-row actions trigger.
    expect(screen.queryByTestId("credential-actions-b-openai-key")).not.toBeInTheDocument();
    expect(screen.queryByTestId("credential-actions-a-azure-key")).not.toBeInTheDocument();
  });

  describe("URL state", () => {
    const manyCredentials: CredentialItem[] = Array.from({ length: 30 }, (_, index) => ({
      credential_name: `cred-${String(index + 1).padStart(2, "0")}`,
      credential_values: {},
      credential_info: { custom_llm_provider: "openai" },
    }));

    const credentialNamesInOrder = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((row) => within(row).getAllByRole("cell")[0].textContent);

    const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => {
      const lastCall = onUrlUpdate.mock.calls.at(-1);
      if (!lastCall) throw new Error("expected a URL update");
      return lastCall[0].searchParams;
    };

    it("sorts in the direction named in the URL", () => {
      renderWithProviders(<CredentialsTable {...defaultProps} />, {
        searchParams: "?credentials_sort_order=desc",
      });

      expect(credentialNamesInOrder()).toEqual(["b-openai-key", "a-azure-key"]);
    });

    it("writes a sort toggle to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<CredentialsTable {...defaultProps} />, { onUrlUpdate });

      await user.click(screen.getByTestId("sort-header-credential_name"));

      await waitFor(() => expect(lastUrl(onUrlUpdate).get("credentials_sort_order")).toBe("desc"));
      expect(lastUrl(onUrlUpdate).has("credentials_sort_by")).toBe(false);
      expect(credentialNamesInOrder()[0]).toBe("b-openai-key");
    });

    it("shows the page named in the URL", () => {
      renderWithProviders(<CredentialsTable {...defaultProps} credentials={manyCredentials} />, {
        searchParams: "?credentials_page=2",
      });

      expect(credentialNamesInOrder()).toEqual(["cred-26", "cred-27", "cred-28", "cred-29", "cred-30"]);
    });

    it("writes a page change to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<CredentialsTable {...defaultProps} credentials={manyCredentials} />, { onUrlUpdate });

      await user.click(screen.getByTestId("pagination-next"));

      await waitFor(() => expect(lastUrl(onUrlUpdate).get("credentials_page")).toBe("2"));
      expect(credentialNamesInOrder()[0]).toBe("cred-26");
    });
  });
});

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { useFormContext } from "react-hook-form";
import { Providers } from "../provider_info_helpers";
import { GITHUB_COPILOT_AUTH_TYPE_KEY } from "./provider_auth_types";
import type { MountedFormValues } from "../common_components/MountedFormField";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import ProviderSpecificFields from "./provider_specific_fields";

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    getProviderCreateMetadata: vi.fn().mockResolvedValue([
      {
        provider: "GITHUB_COPILOT",
        provider_display_name: Providers.GITHUB_COPILOT,
        litellm_provider: "github_copilot",
        default_model_placeholder: "github_copilot/chat",
        credential_fields: [
          { key: "api_base", label: "API Base", field_type: "text" },
          { key: "api_key", label: "API Key", field_type: "password" },
        ],
      },
    ]),
  };
});

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const GitHubCopilotAuthTypeProbe = () => {
  const { watch } = useFormContext<MountedFormValues>();
  return <output data-testid="github-copilot-auth-type">{String(watch(GITHUB_COPILOT_AUTH_TYPE_KEY) ?? "")}</output>;
};

describe("ProviderSpecificFields", () => {
  it("offers per-user GitHub OAuth as a credential-only model auth type without mounting its fixed value", async () => {
    const onCreateCredential = vi.fn();
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields
            selectedProvider="GITHUB_COPILOT"
            context="model"
            initialAuthTypeId="per_user_oauth"
            onCreateCredential={onCreateCredential}
          />
          <GitHubCopilotAuthTypeProbe />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    expect(await screen.findByRole("combobox", { name: "Auth Type:" })).toHaveTextContent("Per-user GitHub OAuth");
    expect(screen.getByRole("button", { name: "Create credential" })).toBeInTheDocument();
    expect(screen.getByTestId("github-copilot-auth-type")).toBeEmptyDOMElement();
    await userEvent.click(screen.getByRole("button", { name: "Create credential" }));
    expect(onCreateCredential).toHaveBeenCalledWith("per_user_oauth");
  });
});

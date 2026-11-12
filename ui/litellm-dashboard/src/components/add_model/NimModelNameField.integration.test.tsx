import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useFormContext } from "react-hook-form";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import type { MountedFormValues } from "../common_components/MountedFormField";
import { apiClient } from "../networking";
import { getPlaceholder, Providers } from "../provider_info_helpers";
import NimModelNameField from "./NimModelNameField";
import i18n from "@/i18n/i18n";

vi.mock("../networking", () => ({ apiClient: { post: vi.fn() } }));
const catalogModel = "nvidia_nim/catalog-model";
const discoveredModel = "nvidia_nim/discovered-model";

function ConnectionFields() {
  const { register } = useFormContext<MountedFormValues>();
  return (
    <>
      <input aria-label="Credential" {...register("litellm_credential_name")} />
      <input aria-label="API Base" {...register("api_base")} />
      <input aria-label="API Key" {...register("api_key")} />
    </>
  );
}

function Form({
  provider = "NVIDIA_NIM",
  canDiscover = true,
  credentialName = "nim-saved",
  apiKey,
  apiBase,
}: {
  provider?: string;
  canDiscover?: boolean;
  credentialName?: string;
  apiKey?: string;
  apiBase?: string;
} = {}) {
  return (
    <MountedFormHost defaultValues={{ litellm_credential_name: credentialName, api_key: apiKey, api_base: apiBase }}>
      <ConnectionFields />
      <NimModelNameField
        selectedProvider={provider}
        providerModels={[catalogModel]}
        getPlaceholder={getPlaceholder}
        accessToken="proxy-token"
        canDiscover={canDiscover}
      />
    </MountedFormHost>
  );
}

beforeEach(() => vi.resetAllMocks());

describe("NIM model discovery", () => {
  it("uses the selected credential and makes discovered models selectable", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ status: "success", models: [discoveredModel] });
    render(<Form />);
    expect(await screen.findByText(/Models listed by your NIM endpoint/)).toBeInTheDocument();
    expect(apiClient.post).toHaveBeenCalledWith("/credentials/nvidia_nim/models", {
      accessToken: "proxy-token",
      body: { credential_name: "nim-saved", credential_values: undefined },
      signal: expect.any(AbortSignal),
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Select models" }));
    expect(screen.getByRole("option", { name: discoveredModel })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: catalogModel })).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Custom Model Name (Enter below)" })).toBeInTheDocument();
  });

  it("requires explicit refresh for inline settings and never sends keys while editing", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ status: "success", models: [discoveredModel] });
    vi.useFakeTimers();
    try {
      render(<Form credentialName="" apiKey="inline-key" apiBase="https://nim.example/v1" />);
      await act(() => vi.advanceTimersByTimeAsync(1000));
      expect(apiClient.post).not.toHaveBeenCalled();
      fireEvent.click(screen.getByRole("button", { name: "Refresh NIM models" }));
      await act(() => vi.advanceTimersByTimeAsync(1000));
      expect(apiClient.post).toHaveBeenCalledTimes(1);
      expect(apiClient.post).toHaveBeenLastCalledWith("/credentials/nvidia_nim/models", {
        accessToken: "proxy-token",
        body: {
          credential_name: undefined,
          credential_values: { api_key: "inline-key", api_base: "https://nim.example/v1" },
        },
        signal: expect.any(AbortSignal),
      });
      fireEvent.change(screen.getByRole("textbox", { name: "API Base" }), {
        target: { value: "https://partial.example/v1" },
      });
      await act(() => vi.advanceTimersByTimeAsync(1000));
      expect(apiClient.post).toHaveBeenCalledTimes(1);
      expect(screen.getByText(/Enter your connection settings/)).toBeInTheDocument();
      fireEvent.change(screen.getByRole("textbox", { name: "API Key" }), { target: { value: "new-key" } });
      await act(() => vi.advanceTimersByTimeAsync(1000));
      expect(apiClient.post).toHaveBeenCalledTimes(1);
      fireEvent.click(screen.getByRole("button", { name: "Refresh NIM models" }));
      await act(() => vi.advanceTimersByTimeAsync(1000));
      expect(apiClient.post).toHaveBeenCalledTimes(2);
      expect(apiClient.post).toHaveBeenLastCalledWith("/credentials/nvidia_nim/models", {
        accessToken: "proxy-token",
        body: {
          credential_name: undefined,
          credential_values: { api_key: "new-key", api_base: "https://partial.example/v1" },
        },
        signal: expect.any(AbortSignal),
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("translates discovery status and refresh controls in the selected language", async () => {
    const previousLanguage = i18n.language;
    i18n.addResourceBundle("es", "translation", {
      addModel: {
        nimModelDiscovery: {
          loading: "Translated loading",
          unavailable: "Translated unavailable",
          refresh: "Translated refresh",
        },
      },
    });
    try {
      await act(() => i18n.changeLanguage("es"));
      vi.mocked(apiClient.post).mockResolvedValue({ status: "unavailable", models: [] });
      render(<Form />);
      expect(screen.getByText("Translated loading")).toBeInTheDocument();
      expect(await screen.findByText("Translated unavailable")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Translated refresh" })).toBeEnabled();
    } finally {
      await act(() => i18n.changeLanguage(previousLanguage));
      i18n.removeResourceBundle("es", "translation");
    }
  });

  it("keeps the catalog usable when upstream discovery is unavailable", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ status: "unavailable", models: [] });
    render(<Form />);
    expect(await screen.findByText(/Could not refresh NIM models/)).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Select models" }));
    expect(screen.getByRole("option", { name: catalogModel })).toBeInTheDocument();
  });

  it("ignores a late response for a previous credential", async () => {
    let completeOld: (value: unknown) => void = () => {};
    vi.mocked(apiClient.post)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            completeOld = resolve;
          }),
      )
      .mockResolvedValueOnce({ status: "success", models: [discoveredModel] });
    render(<Form />);
    await waitFor(() => expect(apiClient.post).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByRole("textbox", { name: "Credential" }), { target: { value: "nim-new" } });
    expect(await screen.findByText(/Models listed by your NIM endpoint/)).toBeInTheDocument();
    completeOld({ status: "success", models: ["nvidia_nim/old-credential-model"] });
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Select models" }));
    expect(screen.getByRole("option", { name: discoveredModel })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "nvidia_nim/old-credential-model" })).not.toBeInTheDocument();
  });

  it.each([{ provider: Providers.OpenAI }, { canDiscover: false }])(
    "leaves other providers and roles on the catalog",
    async (props) => {
      vi.useFakeTimers();
      try {
        render(<Form {...props} />);
        await act(() => vi.advanceTimersByTimeAsync(600));
        expect(screen.queryByRole("button", { name: "Refresh NIM models" })).not.toBeInTheDocument();
        expect(apiClient.post).not.toHaveBeenCalled();
      } finally {
        vi.useRealTimers();
      }
    },
  );
});

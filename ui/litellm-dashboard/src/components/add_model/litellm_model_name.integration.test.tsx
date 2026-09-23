import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import { getPlaceholder, Providers } from "../provider_info_helpers";
import LiteLLMModelNameField from "./litellm_model_name";

describe("NanoGPT model selection", () => {
  it.each(["NANOGPT", "nano-gpt", Providers.NANOGPT])(
    "offers and selects all models for %s without a static provider catalog",
    async (provider) => {
      render(
        <MountedFormHost>
          <LiteLLMModelNameField selectedProvider={provider} providerModels={[]} getPlaceholder={getPlaceholder} />
        </MountedFormHost>,
      );
      await userEvent.click(screen.getByRole("combobox"));
      await userEvent.click(await screen.findByRole("option", { name: "All NanoGPT Models (Wildcard)" }));
      expect(screen.getByLabelText("All NanoGPT Models (Wildcard)")).toBeInTheDocument();
    },
  );
});

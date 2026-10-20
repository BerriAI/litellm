import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import { getPlaceholder, Providers } from "../provider_info_helpers";
import LiteLLMModelNameField from "./litellm_model_name";

describe("MiniMax saved model selection", () => {
  it.each(["minimax/MiniMax-M2.1", "minimax/my-custom-model"])("keeps %s selected when ordinary choices change", (savedModel) => {
    render(
      <MountedFormHost defaultValues={{ model: [savedModel] }}>
        <LiteLLMModelNameField
          selectedProvider={Providers.MiniMax}
          providerModels={["minimax/MiniMax-M3.1-Flash", "minimax/MiniMax-M3"]}
          getPlaceholder={getPlaceholder}
        />
      </MountedFormHost>,
    );
    expect(screen.getByText(savedModel)).toBeInTheDocument();
  });
});

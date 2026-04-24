import { render } from "@testing-library/react";
import { FormProvider, useForm } from "react-hook-form";
import { describe, expect, it } from "vitest";
import { getPlaceholder, Providers } from "../provider_info_helpers";
import LiteLLMModelNameField from "./litellm_model_name";

function Wrapper({ children }: { children: React.ReactNode }) {
  const form = useForm({ defaultValues: {} });
  return <FormProvider {...form}>{children}</FormProvider>;
}

describe("LitellmModelNameField", () => {
  it("should render", () => {
    const { getByText } = render(
      <Wrapper>
        <LiteLLMModelNameField
          selectedProvider={Providers.OpenAI}
          providerModels={[]}
          getPlaceholder={getPlaceholder}
        />
      </Wrapper>,
    );
    expect(getByText("LiteLLM Model Name(s)")).toBeInTheDocument();
  });

  it("should show Azure placeholder as 'my-deployment'", () => {
    const { getByPlaceholderText, queryByPlaceholderText } = render(
      <Wrapper>
        <LiteLLMModelNameField
          selectedProvider={Providers.Azure}
          providerModels={[]}
          getPlaceholder={getPlaceholder}
        />
      </Wrapper>,
    );
    expect(getByPlaceholderText("my-deployment")).toBeInTheDocument();
    expect(queryByPlaceholderText("gpt-3.5-turbo")).toBeNull();
  });
});

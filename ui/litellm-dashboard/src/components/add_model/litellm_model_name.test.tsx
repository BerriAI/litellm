import { render } from "@testing-library/react";
import { Form } from "antd";
import { describe, expect, it } from "vitest";
import { getPlaceholder, Providers } from "../provider_info_helpers";
import LiteLLMModelNameField from "./litellm_model_name";

describe("LitellmModelNameField", () => {
  it("should render", () => {
    const { getByText } = render(
      <Form>
        <LiteLLMModelNameField
          selectedProvider={Providers.OpenAI}
          providerModels={[]}
          getPlaceholder={getPlaceholder}
        />
      </Form>,
    );
    expect(getByText("LiteLLM Model Name(s)")).toBeInTheDocument();
  });

  it("should show Azure placeholder as 'my-deployment'", () => {
    const { getByPlaceholderText, queryByPlaceholderText } = render(
      <Form>
        <LiteLLMModelNameField selectedProvider={Providers.Azure} providerModels={[]} getPlaceholder={getPlaceholder} />
      </Form>,
    );
    expect(getByPlaceholderText("my-deployment")).toBeInTheDocument();
    expect(queryByPlaceholderText("gpt-3.5-turbo")).toBeNull();
  });
});

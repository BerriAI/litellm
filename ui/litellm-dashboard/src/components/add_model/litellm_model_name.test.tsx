import { fireEvent, render, screen } from "@testing-library/react";
import { Form } from "antd";
import { describe, expect, it } from "vitest";
import { getPlaceholder, Providers } from "../provider_info_helpers";
import LiteLLMModelNameField from "./litellm_model_name";

const ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-opus-4-20250514"];

const renderWithCustomSelected = (providerModels: string[]) => {
  return render(
    <Form initialValues={{ model: ["custom"] }}>
      <LiteLLMModelNameField
        selectedProvider={Providers.Anthropic}
        providerModels={providerModels}
        getPlaceholder={getPlaceholder}
      />
    </Form>,
  );
};

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

  it("shows a warning Alert when the user types a misnamed custom model like 'claude'", () => {
    renderWithCustomSelected(ANTHROPIC_MODELS);

    const input = screen.getByPlaceholderText("Enter custom model name");
    fireEvent.change(input, { target: { value: "claude" } });

    const warning = screen.getByTestId("model-name-warning");
    expect(warning).toBeInTheDocument();
    expect(warning).toHaveTextContent(`"claude" doesn't match a known Anthropic model`);
    expect(warning).toHaveTextContent("claude-3-5-sonnet-20240620");
  });

  it("does not show a warning when the typed custom name matches a known model exactly", () => {
    renderWithCustomSelected(ANTHROPIC_MODELS);

    const input = screen.getByPlaceholderText("Enter custom model name");
    fireEvent.change(input, { target: { value: "claude-3-opus-20240229" } });

    expect(screen.queryByTestId("model-name-warning")).toBeNull();
  });

  it("does not show a warning when the typed name doesn't look like any known model", () => {
    renderWithCustomSelected(ANTHROPIC_MODELS);

    const input = screen.getByPlaceholderText("Enter custom model name");
    fireEvent.change(input, { target: { value: "my-private-finetune" } });

    expect(screen.queryByTestId("model-name-warning")).toBeNull();
  });
});

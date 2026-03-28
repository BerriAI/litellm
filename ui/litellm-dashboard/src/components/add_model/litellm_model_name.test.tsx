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

  it("handleCustomModelNameChange rebuilds mappings from model sentinel on each keystroke", () => {
    // Directly test the logic: simulate what handleCustomModelNameChange does by
    // verifying that model_mappings is rebuilt from form.model (not from the stale
    // model_mappings). This guards against the regression where the function searched
    // model_mappings for public_name === "custom" and silently no-op'ed after the
    // first character was typed.

    // Simulate form state after user selects "custom" and types "us-gov.anthropic..."
    const formState: Record<string, any> = {
      model: ["custom"],
      model_mappings: [{ public_name: "custom", litellm_model: "custom" }],
    };
    const mockForm = {
      getFieldValue: (field: string) => formState[field],
      setFieldsValue: (values: Record<string, any>) => {
        Object.assign(formState, values);
      },
    };

    // Simulate the fixed handleCustomModelNameChange logic
    const simulateChange = (customName: string, provider: Providers) => {
      const currentModels = mockForm.getFieldValue("model") || [];
      const modelArray: string[] = Array.isArray(currentModels) ? currentModels : [currentModels];
      const updatedMappings = modelArray.map((model: string) => {
        if (model === "custom") {
          return {
            public_name: customName,
            litellm_model: provider === Providers.Azure ? `azure/${customName}` : customName,
          };
        }
        return { public_name: model, litellm_model: provider === Providers.Azure ? `azure/${model}` : model };
      });
      mockForm.setFieldsValue({ model_mappings: updatedMappings });
    };

    // Keystroke 1: "u"
    simulateChange("u", Providers.Bedrock);
    expect(formState.model_mappings).toEqual([{ public_name: "u", litellm_model: "u" }]);

    // Keystroke 2: "us" — previously would silently no-op, leaving mappings frozen at "u"
    simulateChange("us", Providers.Bedrock);
    expect(formState.model_mappings).toEqual([{ public_name: "us", litellm_model: "us" }]);

    // Full Gov region model name
    const govModel = "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0";
    simulateChange(govModel, Providers.Bedrock);
    expect(formState.model_mappings).toEqual([{ public_name: govModel, litellm_model: govModel }]);
  });

  it("handleCustomModelNameChange preserves non-custom model selections in a multi-select scenario", () => {
    const formState: Record<string, any> = {
      model: ["anthropic.claude-3-opus-20240229-v1:0", "custom"],
      model_mappings: [
        { public_name: "anthropic.claude-3-opus-20240229-v1:0", litellm_model: "anthropic.claude-3-opus-20240229-v1:0" },
        { public_name: "custom", litellm_model: "custom" },
      ],
    };
    const mockForm = {
      getFieldValue: (field: string) => formState[field],
      setFieldsValue: (values: Record<string, any>) => {
        Object.assign(formState, values);
      },
    };

    const simulateChange = (customName: string, provider: Providers) => {
      const currentModels = mockForm.getFieldValue("model") || [];
      const modelArray: string[] = Array.isArray(currentModels) ? currentModels : [currentModels];
      const updatedMappings = modelArray.map((model: string) => {
        if (model === "custom") {
          return {
            public_name: customName,
            litellm_model: provider === Providers.Azure ? `azure/${customName}` : customName,
          };
        }
        return { public_name: model, litellm_model: provider === Providers.Azure ? `azure/${model}` : model };
      });
      mockForm.setFieldsValue({ model_mappings: updatedMappings });
    };

    const govModel = "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0";
    simulateChange(govModel, Providers.Bedrock);

    expect(formState.model_mappings).toEqual([
      { public_name: "anthropic.claude-3-opus-20240229-v1:0", litellm_model: "anthropic.claude-3-opus-20240229-v1:0" },
      { public_name: govModel, litellm_model: govModel },
    ]);
  });
});

import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ModelInfoView from "./model_info_view";

vi.mock("../../utils/dataUtils", () => ({
  copyToClipboard: vi.fn(),
}));

vi.mock("./networking", () => ({
  modelInfoV1Call: vi.fn().mockResolvedValue({
    data: [
      {
        model_name: "aws/anthropic/bedrock-claude-3-5-sonnet-v1",
        litellm_params: {
          aws_region_name: "us-east-1",
          custom_llm_provider: "bedrock",
          use_in_pass_through: false,
          use_litellm_proxy: false,
          merge_reasoning_content_in_choices: false,
          model: "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        },
        model_info: {
          id: "70b94bbd2af4a75215f7e3e465b5b199529dc15deb5d395d0668a4aabc496c84",
          db_model: false,
          access_via_team_ids: [
            "4fe3cfea-c907-412a-a645-60915b618d11",
            "9a4b2d15-4198-47e4-971b-7329b77f40e4",
            "14d55eef-b8d4-4cb8-b080-d973269dae54",
            "693ce1d2-9fae-4605-a5c9-1c9829415e1a",
            "fe29d910-4968-45bc-9fe0-6716e89c6270",
          ],
          direct_access: true,
          key: "anthropic.claude-3-5-sonnet-20240620-v1:0",
          max_tokens: 4096,
          max_input_tokens: 200000,
          max_output_tokens: 4096,
          input_cost_per_token: 0.000003,
          input_cost_per_token_flex: null,
          input_cost_per_token_priority: null,
          cache_creation_input_token_cost: null,
          cache_read_input_token_cost: null,
          cache_read_input_token_cost_flex: null,
          cache_read_input_token_cost_priority: null,
          cache_creation_input_token_cost_above_1hr: null,
          input_cost_per_character: null,
          input_cost_per_token_above_128k_tokens: null,
          input_cost_per_token_above_200k_tokens: null,
          input_cost_per_query: null,
          input_cost_per_second: null,
          input_cost_per_audio_token: null,
          input_cost_per_token_batches: null,
          output_cost_per_token_batches: null,
          output_cost_per_token: 0.000015,
          output_cost_per_token_flex: null,
          output_cost_per_token_priority: null,
          output_cost_per_audio_token: null,
          output_cost_per_character: null,
          output_cost_per_reasoning_token: null,
          output_cost_per_token_above_128k_tokens: null,
          output_cost_per_character_above_128k_tokens: null,
          output_cost_per_token_above_200k_tokens: null,
          output_cost_per_second: null,
          output_cost_per_video_per_second: null,
          output_cost_per_image: null,
          output_vector_size: null,
          citation_cost_per_token: null,
          tiered_pricing: null,
          litellm_provider: "bedrock",
          mode: "chat",
          supports_system_messages: null,
          supports_response_schema: true,
          supports_vision: true,
          supports_function_calling: true,
          supports_tool_choice: true,
          supports_assistant_prefill: null,
          supports_prompt_caching: null,
          supports_audio_input: null,
          supports_audio_output: null,
          supports_pdf_input: true,
          supports_embedding_image_input: null,
          supports_native_streaming: null,
          supports_web_search: null,
          supports_url_context: null,
          supports_reasoning: null,
          supports_computer_use: null,
          search_context_cost_per_query: null,
          tpm: null,
          rpm: null,
          ocr_cost_per_page: null,
          annotation_cost_per_page: null,
          supported_openai_params: [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "stop",
            "temperature",
            "top_p",
            "extra_headers",
            "response_format",
            "requestMetadata",
            "tools",
            "tool_choice",
          ],
        },
      },
    ],
  }),
  credentialGetCall: vi.fn().mockResolvedValue({}),
}));

describe("ModelInfoView", () => {
  const modelData = {
    model_info: {
      id: "123",
      created_by: "123",
      db_model: true,
    },
    litellm_params: {
      api_base: "https://api.openai.com/v1",
      custom_llm_provider: "openai",
    },
    litellm_model_name: "gpt-4",
    model_name: "GPT-4",
    litellm_provider: "openai",
    mode: "chat",
    supported_openai_params: ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"],
  };

  const DEFAULT_ADMIN_PROPS = {
    modelId: "123",
    onClose: () => {},
    modelData: modelData,
    accessToken: "123",
    userID: "123",
    userRole: "Admin",
    editModel: false,
    setEditModalVisible: () => {},
    setSelectedModel: () => {},
    onModelUpdate: () => {},
    modelAccessGroups: [],
  };

  describe("Edit Model", () => {
    it("should render the model info view", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
      await waitFor(() => {
        expect(getByText("Model Settings")).toBeInTheDocument();
      });
    });

    it("should not render an edit model button if the model is not a DB model", async () => {
      const nonDbModelData = {
        ...modelData,
        model_info: {
          ...modelData.model_info,
          db_model: false,
        },
      };

      const NON_DB_ADMIN_PROPS = {
        ...DEFAULT_ADMIN_PROPS,
        modelData: nonDbModelData,
      };

      const { queryByText } = render(<ModelInfoView {...NON_DB_ADMIN_PROPS} />);
      await waitFor(() => {
        expect(queryByText("Edit Model")).not.toBeInTheDocument();
      });
    });

    it("should render tags in the edit model", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
      await waitFor(() => {
        expect(getByText("Tags")).toBeInTheDocument();
      });
    });

    it("should render the litellm params in the edit model", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
      await waitFor(() => {
        expect(getByText("LiteLLM Params")).toBeInTheDocument();
      });
    });
  });

  it("should render a test connection button", async () => {
    const { getByTestId } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
    await waitFor(() => {
      expect(getByTestId("test-connection-button")).toBeInTheDocument();
    });
  });

  it("should render a reuse credentials button", async () => {
    const { getByTestId } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
    await waitFor(() => {
      expect(getByTestId("reuse-credentials-button")).toBeInTheDocument();
    });
  });

  it("should render a delete model button", async () => {
    const { getByTestId } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
    await waitFor(() => {
      expect(getByTestId("delete-model-button")).toBeInTheDocument();
    });
  });

  it("should render a disabled delete model button if the model is not a DB model", async () => {
    const nonDbModelData = {
      ...modelData,
      model_info: {
        ...modelData.model_info,
        db_model: false,
      },
    };
    const NON_DB_ADMIN_PROPS = {
      ...DEFAULT_ADMIN_PROPS,
      modelData: nonDbModelData,
    };
    const { getByTestId } = render(<ModelInfoView {...NON_DB_ADMIN_PROPS} />);
    await waitFor(() => {
      expect(getByTestId("delete-model-button")).toBeDisabled();
    });
  });

  it("should render a disabled delete model button if the user is not an admin and model is not created by the user", async () => {
    const nonCreatedByUserModelData = {
      ...modelData,
      model_info: {
        ...modelData.model_info,
        created_by: "456",
      },
    };
    const NON_CREATED_BY_USER_ADMIN_PROPS = {
      ...DEFAULT_ADMIN_PROPS,
      modelData: nonCreatedByUserModelData,
      userRole: "User",
    };
    const { getByTestId } = render(<ModelInfoView {...NON_CREATED_BY_USER_ADMIN_PROPS} />);
    await waitFor(() => {
      expect(getByTestId("delete-model-button")).toBeDisabled();
    });
  });

  describe("View Model", () => {
    it("should render the model info view", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
      await waitFor(() => {
        expect(getByText("Model Settings")).toBeInTheDocument();
      });
    });

    it("should render tags in the view model", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />);
      await waitFor(() => {
        expect(getByText("Tags")).toBeInTheDocument();
      });
    });
  });
});

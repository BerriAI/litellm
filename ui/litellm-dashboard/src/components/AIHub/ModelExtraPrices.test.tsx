import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModelExtraPrices } from "./ModelExtraPrices";

describe("ModelExtraPrices", () => {
  it("lists every USD price besides the per-token pair, sorted by name, with token prices per 1M tokens", () => {
    render(
      <ModelExtraPrices
        model={{
          model_group: "image-model",
          input_cost_per_token: 0.000001,
          output_cost_per_token: 0.000002,
          output_cost_per_image: 0.04,
          input_cost_per_token_priority: 0.0000025,
          output_cost_per_image_4K: 0.16,
          cache_read_input_token_cost: 0.0000005,
          input_dbu_cost_per_token: 0.000142858,
          cost_center: 4021,
          output_cost_per_video_per_second: "free",
          search_context_cost_per_query: { search_context_size_low: 0.01 },
          max_input_tokens: 128000,
        }}
      />,
    );
    const terms = screen.getAllByRole("term").map((term) => term.textContent);
    const definitions = screen.getAllByRole("definition").map((definition) => definition.textContent);
    expect(terms.map((term, index) => [term, definitions[index]])).toEqual([
      ["cache_read_input_token_cost", "$0.5 per 1M tokens"],
      ["input_cost_per_token_priority", "$2.5 per 1M tokens"],
      ["output_cost_per_image", "$0.04"],
      ["output_cost_per_image_4K", "$0.16"],
    ]);
  });

  it("renders nothing when the model lists only the per-token prices", () => {
    const { container } = render(
      <ModelExtraPrices
        model={{ model_group: "chat-model", input_cost_per_token: 0.000001, output_cost_per_token: 0.000002 }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

import { render, screen } from "@testing-library/react";
import { Form } from "antd";
import { describe, expect, it } from "vitest";
import ConditionalPublicModelName from "./conditional_public_model_name";

describe("ConditionalPublicModelName", () => {
  it("should render", () => {
    render(
      <Form
        initialValues={{
          model: ["gpt-4"],
          model_mappings: [
            {
              public_name: "gpt-4",
              litellm_model: "gpt-4",
            },
          ],
        }}
      >
        <ConditionalPublicModelName />
      </Form>,
    );

    expect(screen.getByText("Model Mappings")).toBeInTheDocument();
    expect(screen.getByText("Public Model Name")).toBeInTheDocument();
    expect(screen.getByText("LiteLLM Model Name")).toBeInTheDocument();
  });
});

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import ConditionalPublicModelName from "./conditional_public_model_name";

describe("ConditionalPublicModelName", () => {
  it("should render", () => {
    render(
      <MountedFormHost
        defaultValues={{
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
      </MountedFormHost>,
    );

    expect(screen.getByText("Model Mappings")).toBeInTheDocument();
    expect(screen.getByText("Public Model Name")).toBeInTheDocument();
    expect(screen.getByText("LiteLLM Model Name")).toBeInTheDocument();
  });
});

import { render, screen } from "@testing-library/react";
import React, { useEffect, useRef } from "react";
import { useFormContext, useWatch } from "react-hook-form";
import { describe, expect, it } from "vitest";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import type { MountedFormValues } from "../common_components/MountedFormField";
import ConditionalPublicModelName from "./conditional_public_model_name";

const WRITE_BUDGET = 20;

const LoopGuard: React.FC = () => {
  const form = useFormContext<MountedFormValues>();
  const mappings = useWatch({ control: form.control, name: "model_mappings" });
  const writes = useRef(0);

  useEffect(() => {
    writes.current += 1;
    if (writes.current > WRITE_BUDGET) {
      throw new Error(`model_mappings changed ${WRITE_BUDGET}+ times: the mapping effects are looping`);
    }
  }, [mappings]);

  return null;
};

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

  it("settles after rewriting the custom placeholder mapping to the entered model name", () => {
    render(
      <MountedFormHost
        defaultValues={{
          model: ["custom"],
          custom_model_name: "my-custom-model",
          model_mappings: [
            {
              public_name: "custom",
              litellm_model: "custom",
            },
          ],
        }}
      >
        <ConditionalPublicModelName />
        <LoopGuard />
      </MountedFormHost>,
    );

    expect(screen.getByDisplayValue("my-custom-model")).toBeInTheDocument();
    expect(screen.getByText("my-custom-model")).toBeInTheDocument();
    expect(screen.queryByDisplayValue("custom")).not.toBeInTheDocument();
  });
});

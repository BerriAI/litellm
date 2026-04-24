import { render, screen } from "@testing-library/react";
import { FormProvider, useForm } from "react-hook-form";
import { describe, expect, it } from "vitest";
import ConditionalPublicModelName from "./conditional_public_model_name";

function Wrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  const form = useForm({
    defaultValues: {
      model: ["gpt-4"],
      model_mappings: [
        {
          public_name: "gpt-4",
          litellm_model: "gpt-4",
        },
      ],
    },
  });
  return <FormProvider {...form}>{children}</FormProvider>;
}

describe("ConditionalPublicModelName", () => {
  it("should render", () => {
    render(
      <Wrapper>
        <ConditionalPublicModelName />
      </Wrapper>,
    );

    expect(screen.getByText("Model Mappings")).toBeInTheDocument();
    expect(screen.getByText("Public Model Name")).toBeInTheDocument();
    expect(screen.getByText("LiteLLM Model Name")).toBeInTheDocument();
  });
});

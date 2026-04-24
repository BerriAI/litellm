import { act, fireEvent, render, waitFor } from "@testing-library/react";
import { FormProvider, useForm } from "react-hook-form";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AdvancedSettings from "./advanced_settings";

function Wrapper({ children }: { children: React.ReactNode }) {
  const form = useForm({ defaultValues: {} });
  return <FormProvider {...form}>{children}</FormProvider>;
}

describe("AdvancedSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    render(
      <Wrapper>
        <AdvancedSettings
          showAdvancedSettings={true}
          setShowAdvancedSettings={() => {}}
          guardrailsList={[]}
          tagsList={{}}
          accessToken="test-token"
        />
      </Wrapper>,
    );
  });

  it("should render tags list", async () => {
    const { getByText } = render(
      <Wrapper>
        <AdvancedSettings
          showAdvancedSettings={true}
          setShowAdvancedSettings={() => {}}
          guardrailsList={[]}
          tagsList={{}}
          accessToken="test-token"
        />
      </Wrapper>,
    );
    fireEvent.click(getByText("Advanced Settings"));
    await waitFor(() => {
      expect(getByText("Tags")).toBeInTheDocument();
    });
  });

  it("should render the litellm params", async () => {
    const { getByText } = render(
      <Wrapper>
        <AdvancedSettings
          showAdvancedSettings={true}
          setShowAdvancedSettings={() => {}}
          guardrailsList={[]}
          tagsList={{}}
          accessToken="test-token"
        />
      </Wrapper>,
    );
    act(() => {
      fireEvent.click(getByText("Advanced Settings"));
    });
    await waitFor(() => {
      expect(getByText("LiteLLM Params")).toBeInTheDocument();
    });
  });
});

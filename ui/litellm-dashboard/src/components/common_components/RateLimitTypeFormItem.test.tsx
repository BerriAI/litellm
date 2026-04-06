import { renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { Form } from "antd";
import React from "react";
import { RateLimitTypeFormItem } from "./RateLimitTypeFormItem";

const Wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Form>{children}</Form>
);

describe("RateLimitTypeFormItem", () => {
  it("should render", () => {
    renderWithProviders(
      <Wrapper>
        <RateLimitTypeFormItem type="tpm" name="tpm_type" />
      </Wrapper>
    );
    expect(screen.getByText(/TPM Rate Limit Type/)).toBeInTheDocument();
  });

  it("should display TPM label for tpm type", () => {
    renderWithProviders(
      <Wrapper>
        <RateLimitTypeFormItem type="tpm" name="tpm_type" />
      </Wrapper>
    );
    expect(screen.getByText(/TPM Rate Limit Type/)).toBeInTheDocument();
  });

  it("should display RPM label for rpm type", () => {
    renderWithProviders(
      <Wrapper>
        <RateLimitTypeFormItem type="rpm" name="rpm_type" />
      </Wrapper>
    );
    expect(screen.getByText(/RPM Rate Limit Type/)).toBeInTheDocument();
  });

  it("should show the select placeholder by default", () => {
    renderWithProviders(
      <Wrapper>
        <RateLimitTypeFormItem type="tpm" name="tpm_type" />
      </Wrapper>
    );
    expect(screen.getByText("Select rate limit type")).toBeInTheDocument();
  });

  it("should call onChange when provided", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(
      <Wrapper>
        <RateLimitTypeFormItem type="tpm" name="tpm_type" onChange={onChange} />
      </Wrapper>
    );
    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByText("Guaranteed throughput"));
    expect(onChange).toHaveBeenCalledWith("guaranteed_throughput");
  });
});

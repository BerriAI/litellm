import { renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { RateLimitTypeFormItem } from "./RateLimitTypeFormItem";

describe("RateLimitTypeFormItem", () => {
  it("should render", () => {
    renderWithProviders(<RateLimitTypeFormItem type="tpm" name="tpm_type" />);
    expect(screen.getByText(/TPM Rate Limit Type/)).toBeInTheDocument();
  });

  it("should display TPM label for tpm type", () => {
    renderWithProviders(<RateLimitTypeFormItem type="tpm" name="tpm_type" />);
    expect(screen.getByText(/TPM Rate Limit Type/)).toBeInTheDocument();
  });

  it("should display RPM label for rpm type", () => {
    renderWithProviders(<RateLimitTypeFormItem type="rpm" name="rpm_type" />);
    expect(screen.getByText(/RPM Rate Limit Type/)).toBeInTheDocument();
  });

  it("should show the select placeholder by default", () => {
    renderWithProviders(<RateLimitTypeFormItem type="tpm" name="tpm_type" />);
    expect(screen.getByText("Select rate limit type")).toBeInTheDocument();
  });

  it("should call onChange when provided", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<RateLimitTypeFormItem type="tpm" name="tpm_type" onChange={onChange} />);
    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByText("Guaranteed throughput"));
    expect(onChange).toHaveBeenCalledWith("guaranteed_throughput");
  });
});

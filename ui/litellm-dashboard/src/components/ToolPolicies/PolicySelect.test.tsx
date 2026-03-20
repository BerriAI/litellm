import { renderWithProviders, screen } from "../../../tests/test-utils";
import { vi } from "vitest";
import {
  PolicySelect,
  policyStyle,
  INPUT_POLICY_OPTIONS,
  OUTPUT_POLICY_OPTIONS,
} from "./PolicySelect";

describe("policyStyle", () => {
  it("should return the matching option for a known policy", () => {
    expect(policyStyle("trusted")).toEqual(INPUT_POLICY_OPTIONS[1]);
  });

  it("should return the matching option for blocked", () => {
    expect(policyStyle("blocked")).toEqual(INPUT_POLICY_OPTIONS[2]);
  });

  it("should return the first option as fallback for unknown policy", () => {
    expect(policyStyle("unknown")).toEqual(INPUT_POLICY_OPTIONS[0]);
  });
});

describe("PolicySelect", () => {
  it("should render", () => {
    renderWithProviders(
      <PolicySelect
        value="untrusted"
        toolName="test-tool"
        saving={false}
        onChange={vi.fn()}
      />
    );
    expect(screen.getByText("untrusted")).toBeInTheDocument();
  });

  it("should show the current policy value", () => {
    renderWithProviders(
      <PolicySelect
        value="trusted"
        toolName="test-tool"
        saving={false}
        onChange={vi.fn()}
      />
    );
    expect(screen.getByText("trusted")).toBeInTheDocument();
  });

  it("should be disabled when saving is true", () => {
    renderWithProviders(
      <PolicySelect
        value="untrusted"
        toolName="test-tool"
        saving={true}
        onChange={vi.fn()}
      />
    );
    expect(screen.getByRole("combobox")).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("combobox").closest(".ant-select")).toHaveClass("ant-select-disabled");
  });

  it("should not be disabled when saving is false", () => {
    renderWithProviders(
      <PolicySelect
        value="untrusted"
        toolName="test-tool"
        saving={false}
        onChange={vi.fn()}
      />
    );
    expect(screen.getByRole("combobox").closest(".ant-select")).not.toHaveClass("ant-select-disabled");
  });
});

describe("Policy option constants", () => {
  it("should have 3 input policy options", () => {
    expect(INPUT_POLICY_OPTIONS).toHaveLength(3);
  });

  it("should have 2 output policy options (no blocked)", () => {
    expect(OUTPUT_POLICY_OPTIONS).toHaveLength(2);
    expect(OUTPUT_POLICY_OPTIONS.map((o) => o.value)).toEqual(["untrusted", "trusted"]);
  });
});

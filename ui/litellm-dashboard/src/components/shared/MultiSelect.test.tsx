import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MultiSelect, type MultiSelectOption } from "./MultiSelect";

const OPTIONS: MultiSelectOption[] = [
  { value: "vs-alpha", label: "alpha-kb (vs-alpha)" },
  { value: "vs-beta", label: "beta-kb (vs-beta)", description: "second store" },
];

const renderMultiSelect = (props: Partial<React.ComponentProps<typeof MultiSelect>> = {}) => {
  const onValueChange = vi.fn();
  render(<MultiSelect options={OPTIONS} onValueChange={onValueChange} placeholder="Select stores" {...props} />);
  return { onValueChange, input: screen.getByRole("combobox") };
};

const openPopup = async (input: HTMLElement) => {
  await userEvent.click(input);
  return waitFor(() => {
    const popup = document.querySelector("[data-slot='combobox-content']");
    expect(popup).not.toBeNull();
    return popup as HTMLElement;
  });
};

const stubWidth = (element: Element, width: number) =>
  vi.spyOn(element, "getBoundingClientRect").mockReturnValue({
    width,
    height: 32,
    top: 0,
    left: 0,
    right: width,
    bottom: 32,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);

const CHIPS_WIDTH = 300;
const INPUT_WIDTH = 200;

describe("MultiSelect", () => {
  it("anchors the popup to the chips container rather than the inner input", async () => {
    const { input } = renderMultiSelect();
    const chips = input.closest("[data-slot='combobox-chips']");
    expect(chips).not.toBeNull();
    stubWidth(chips as Element, CHIPS_WIDTH);
    stubWidth(input, INPUT_WIDTH);

    const popup = await openPopup(input);
    const positioner = popup.parentElement as HTMLElement;

    expect(positioner.style.getPropertyValue("--anchor-width")).toBe(`${CHIPS_WIDTH}px`);
  });

  it("reports the selected option values", async () => {
    const { onValueChange, input } = renderMultiSelect();

    await openPopup(input);
    await userEvent.click(screen.getByText("alpha-kb (vs-alpha)"));

    expect(onValueChange).toHaveBeenCalledWith(["vs-alpha"]);
  });

  it("renders a chip per selected value", () => {
    renderMultiSelect({ value: ["vs-alpha", "vs-beta"] });

    expect(screen.getByLabelText("alpha-kb (vs-alpha)")).toBeInTheDocument();
    expect(screen.getByLabelText("beta-kb (vs-beta)")).toBeInTheDocument();
  });

  it("labels an unknown selected value with its raw id", () => {
    renderMultiSelect({ value: ["vs-deleted"] });

    expect(screen.getByLabelText("vs-deleted")).toBeInTheDocument();
  });

  it("offers a typed value only when custom values are allowed", async () => {
    const { onValueChange, input } = renderMultiSelect({ allowCustomValues: true });

    await userEvent.type(input, "vs-typed");
    await userEvent.click(await screen.findByText('Create "vs-typed"'));

    expect(onValueChange).toHaveBeenCalledWith(["vs-typed"]);
  });

  it("does not offer a typed value when custom values are disallowed", async () => {
    const { input } = renderMultiSelect();

    await userEvent.type(input, "vs-typed");

    expect(screen.queryByText('Create "vs-typed"')).not.toBeInTheDocument();
    expect(await screen.findByText("No options found")).toBeInTheDocument();
  });
});

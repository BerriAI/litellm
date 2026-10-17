import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MAX_VISIBLE_OPTIONS, MultiSelect, type MultiSelectOption } from "./MultiSelect";

const OPTIONS: MultiSelectOption[] = [
  { value: "vs-alpha", label: "alpha-kb (vs-alpha)" },
  { value: "vs-beta", label: "beta-kb (vs-beta)", description: "second store" },
];

const LARGE_OPTIONS: MultiSelectOption[] = Array.from({ length: 5000 }, (_, index) => ({
  value: `tag-${index}`,
  label: `tag-${index}`,
  description: "shared spend tag description",
}));

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

  it("caps rendered options and reports the number of matching options", async () => {
    const { input } = renderMultiSelect({ options: LARGE_OPTIONS });

    await openPopup(input);

    expect(screen.getAllByRole("option")).toHaveLength(MAX_VISIBLE_OPTIONS);
    expect(screen.getByText("Showing first 100 of 5000 matches. Type to narrow results.")).toBeInTheDocument();
  });

  it("matches option descriptions by default", async () => {
    const { input } = renderMultiSelect();
    await userEvent.type(input, "second store");

    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(screen.getByRole("option", { name: /beta-kb/ })).toBeInTheDocument();
  });

  it("skips descriptions when searchDescriptions is false", async () => {
    const { input } = renderMultiSelect({ options: LARGE_OPTIONS, searchDescriptions: false });
    await userEvent.type(input, "shared spend");

    expect(screen.queryAllByRole("option")).toHaveLength(0);
    expect(await screen.findByText("No options found")).toBeInTheDocument();
  });

  it("filters by label and updates the match hint", async () => {
    const { input } = renderMultiSelect({ options: LARGE_OPTIONS });
    await userEvent.type(input, "tag-49");

    expect(screen.getAllByRole("option")).toHaveLength(MAX_VISIBLE_OPTIONS);
    expect(screen.getByText("Showing first 100 of 111 matches. Type to narrow results.")).toBeInTheDocument();
  });

  it("shows an under-cap label match without the match hint", async () => {
    const { input } = renderMultiSelect({ options: LARGE_OPTIONS });
    await userEvent.type(input, "tag-4999");

    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(screen.getByRole("option", { name: /tag-4999/ })).toBeInTheDocument();
    expect(screen.queryByText(/Showing first/)).not.toBeInTheDocument();
  });

  it("keeps the custom option available when matching results exceed the cap", async () => {
    const { onValueChange, input } = renderMultiSelect({ options: LARGE_OPTIONS, allowCustomValues: true });
    await userEvent.type(input, "tag");

    await userEvent.click(screen.getByRole("option", { name: 'Create "tag"' }));

    expect(onValueChange).toHaveBeenCalledWith(["tag"]);
  });

  it("renders a selected chip outside the visible option slice", async () => {
    const { input } = renderMultiSelect({ options: LARGE_OPTIONS, value: ["tag-4999"] });

    await openPopup(input);

    expect(screen.getByLabelText("tag-4999")).toBeInTheDocument();
    expect(screen.getAllByRole("option")).toHaveLength(MAX_VISIBLE_OPTIONS);
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

  it("splits a comma-separated custom entry into one value per token", async () => {
    const { onValueChange, input } = renderMultiSelect({ allowCustomValues: true });

    await userEvent.type(input, "udp, kafka ,terraform");
    await userEvent.click(await screen.findByText('Create "udp, kafka ,terraform"'));

    expect(onValueChange).toHaveBeenCalledWith(["udp", "kafka", "terraform"]);
  });

  it("leaves an already selected value that contains a comma alone when a later entry is added", async () => {
    const onValueChange = vi.fn();
    render(
      <MultiSelect
        options={OPTIONS}
        value={["--filter=a,b"]}
        onValueChange={onValueChange}
        allowCustomValues
        placeholder="Select stores"
      />,
    );

    await userEvent.type(screen.getByRole("combobox"), "--verbose");
    await userEvent.click(await screen.findByText('Create "--verbose"'));

    expect(onValueChange).toHaveBeenCalledWith(["--filter=a,b", "--verbose"]);
  });

  it("clears every selection at once", async () => {
    const onValueChange = vi.fn();
    render(
      <MultiSelect
        options={OPTIONS}
        value={["vs-alpha", "vs-beta"]}
        onValueChange={onValueChange}
        placeholder="Select stores"
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Clear all" }));

    expect(onValueChange).toHaveBeenCalledWith([]);
  });

  it("marks a disabled option as disabled and refuses to select it", async () => {
    const { onValueChange, input } = renderMultiSelect({
      options: [OPTIONS[0], { ...OPTIONS[1], disabled: true }],
    });

    await openPopup(input);
    const disabledOption = screen.getByRole("option", { name: /beta-kb/ });
    expect(disabledOption).toHaveAttribute("aria-disabled", "true");

    await userEvent.click(disabledOption);

    expect(onValueChange).not.toHaveBeenCalled();
  });
});

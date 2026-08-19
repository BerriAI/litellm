import { render, screen } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it } from "vitest";

import {
  Field,
  FieldContent,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSeparator,
  FieldSet,
  FieldTitle,
} from "./field";

describe("FieldError", () => {
  it("renders nothing when there are no errors and no children", () => {
    const { container } = render(<FieldError errors={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when every error entry is undefined", () => {
    const { container } = render(<FieldError errors={[undefined, undefined]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders a single message as plain text, not a list", () => {
    render(<FieldError errors={[{ message: "Required" }]} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Required");
    expect(screen.queryByRole("listitem")).not.toBeInTheDocument();
  });

  it("collapses duplicate messages to a single entry", () => {
    render(<FieldError errors={[{ message: "Required" }, { message: "Required" }]} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Required");
    expect(screen.queryByRole("listitem")).not.toBeInTheDocument();
  });

  it("renders distinct messages as a list", () => {
    render(<FieldError errors={[{ message: "Too short" }, { message: "Must be lowercase" }]} />);

    const items = screen.getAllByRole("listitem");
    expect(items.map((item) => item.textContent)).toEqual(["Too short", "Must be lowercase"]);
  });

  it("prefers explicit children over the errors prop", () => {
    render(<FieldError errors={[{ message: "from errors" }]}>from children</FieldError>);

    expect(screen.getByRole("alert")).toHaveTextContent("from children");
    expect(screen.getByRole("alert")).not.toHaveTextContent("from errors");
  });

  it("exposes the message to assistive tech via role=alert", () => {
    render(<FieldError errors={[{ message: "Required" }]} />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
  });
});

describe("Field", () => {
  it("marks itself invalid so descendants can style off it", () => {
    render(
      <Field data-invalid={true}>
        <span>child</span>
      </Field>,
    );

    expect(screen.getByRole("group")).toHaveAttribute("data-invalid", "true");
  });

  it("defaults to vertical orientation", () => {
    render(<Field />);

    expect(screen.getByRole("group")).toHaveAttribute("data-orientation", "vertical");
  });

  it("honours an explicit orientation", () => {
    render(<Field orientation="horizontal" />);

    expect(screen.getByRole("group")).toHaveAttribute("data-orientation", "horizontal");
  });
});

describe("field primitives forward refs to their DOM node", () => {
  it.each([
    ["Field", Field, HTMLDivElement],
    ["FieldContent", FieldContent, HTMLDivElement],
    ["FieldDescription", FieldDescription, HTMLParagraphElement],
    ["FieldGroup", FieldGroup, HTMLDivElement],
    ["FieldLabel", FieldLabel, HTMLLabelElement],
    ["FieldSeparator", FieldSeparator, HTMLDivElement],
    ["FieldTitle", FieldTitle, HTMLDivElement],
  ])("%s", (_name, Component, expected) => {
    const ref = React.createRef<HTMLElement>();
    render(React.createElement(Component as React.ElementType, { ref }));

    expect(ref.current).toBeInstanceOf(expected);
  });

  it("FieldSet and FieldLegend", () => {
    const fieldSet = React.createRef<HTMLFieldSetElement>();
    const legend = React.createRef<HTMLLegendElement>();
    render(
      <FieldSet ref={fieldSet}>
        <FieldLegend ref={legend}>Legend</FieldLegend>
      </FieldSet>,
    );

    expect(fieldSet.current).toBeInstanceOf(HTMLFieldSetElement);
    expect(legend.current).toBeInstanceOf(HTMLLegendElement);
  });

  it("FieldError", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<FieldError ref={ref} errors={[{ message: "Required" }]} />);

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });
});

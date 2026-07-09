import { render } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it } from "vitest";

import { Button } from "./button";
import { Input } from "./input";
import { Label } from "./label";
import { Separator } from "./separator";
import { Skeleton } from "./skeleton";
import { Table, TableBody, TableCaption, TableCell, TableFooter, TableHead, TableHeader, TableRow } from "./table";
import { UiLoadingSpinner } from "./ui-loading-spinner";

describe("ui primitives forward refs to their DOM node", () => {
  it("Button", () => {
    const ref = React.createRef<HTMLButtonElement>();
    render(<Button ref={ref}>ok</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it("Input", () => {
    const ref = React.createRef<HTMLInputElement>();
    render(<Input ref={ref} />);
    expect(ref.current).toBeInstanceOf(HTMLInputElement);
  });

  it("Label", () => {
    const ref = React.createRef<HTMLLabelElement>();
    render(<Label ref={ref}>ok</Label>);
    expect(ref.current).toBeInstanceOf(HTMLLabelElement);
  });

  it("Separator", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<Separator ref={ref} />);
    expect(ref.current).toBeInstanceOf(HTMLElement);
  });

  it("Skeleton", () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<Skeleton ref={ref} />);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it("UiLoadingSpinner", () => {
    const ref = React.createRef<SVGSVGElement>();
    render(<UiLoadingSpinner ref={ref} />);
    expect(ref.current).toBeInstanceOf(SVGSVGElement);
  });

  it("Table family", () => {
    const table = React.createRef<HTMLTableElement>();
    const caption = React.createRef<HTMLTableCaptionElement>();
    const header = React.createRef<HTMLTableSectionElement>();
    const body = React.createRef<HTMLTableSectionElement>();
    const footer = React.createRef<HTMLTableSectionElement>();
    const row = React.createRef<HTMLTableRowElement>();
    const head = React.createRef<HTMLTableCellElement>();
    const cell = React.createRef<HTMLTableCellElement>();

    render(
      <Table ref={table}>
        <TableCaption ref={caption}>caption</TableCaption>
        <TableHeader ref={header}>
          <TableRow ref={row}>
            <TableHead ref={head}>h</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody ref={body}>
          <TableRow>
            <TableCell ref={cell}>d</TableCell>
          </TableRow>
        </TableBody>
        <TableFooter ref={footer}>
          <TableRow>
            <TableCell>f</TableCell>
          </TableRow>
        </TableFooter>
      </Table>,
    );

    expect(table.current).toBeInstanceOf(HTMLTableElement);
    expect(caption.current).toBeInstanceOf(HTMLTableCaptionElement);
    expect(header.current?.tagName).toBe("THEAD");
    expect(body.current?.tagName).toBe("TBODY");
    expect(footer.current?.tagName).toBe("TFOOT");
    expect(row.current).toBeInstanceOf(HTMLTableRowElement);
    expect(head.current?.tagName).toBe("TH");
    expect(cell.current?.tagName).toBe("TD");
  });
});

describe("setupTests ref tripwire", () => {
  it("records a violation when a ref is passed to a plain function component", () => {
    const Plain = (props: React.ComponentPropsWithoutRef<"span">) => <span {...props} />;
    const ref = React.createRef<HTMLSpanElement>();
    render(React.createElement(Plain as never, { ref }));
    const consume = (globalThis as { __consumePendingRefWarnings?: () => string[] }).__consumePendingRefWarnings;
    expect(consume).toBeDefined();
    const violations = consume!();
    expect(violations).toHaveLength(1);
    expect(violations[0]).toContain("Function components cannot be given refs");
  });
});

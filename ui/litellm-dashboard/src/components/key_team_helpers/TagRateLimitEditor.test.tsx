import React, { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { TagRateLimitEditor, TagRateLimitEntry, tagLimitsToRows, tagRowsToLimits } from "./TagRateLimitEditor";

function Harness({ initial = [] as TagRateLimitEntry[], onValue }: { initial?: TagRateLimitEntry[]; onValue?: any }) {
  const [rows, setRows] = useState<TagRateLimitEntry[]>(initial);
  return (
    <TagRateLimitEditor
      value={rows}
      onChange={(next) => {
        setRows(next);
        onValue?.(next);
      }}
    />
  );
}

const rowsWith = (tag: string, rpm: number | null): TagRateLimitEntry[] => [{ id: "r1", tag, rpm_limit: rpm }];

describe("TagRateLimitEditor", () => {
  it("should render one tag and one RPM field per row", () => {
    render(<Harness initial={rowsWith("cell-1", 100)} />);
    expect(screen.getByRole("textbox", { name: "Tag" })).toHaveValue("cell-1");
    expect(screen.getByRole("spinbutton", { name: "RPM limit" })).toHaveValue(100);
  });

  it("should add a row when Add Tag Limit is clicked", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    expect(screen.queryAllByRole("textbox", { name: "Tag" })).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: /add tag limit/i }));

    expect(screen.getAllByRole("textbox", { name: "Tag" })).toHaveLength(1);
  });

  it("should let the user type a tag name", async () => {
    const user = userEvent.setup();
    render(<Harness initial={rowsWith("", null)} />);

    fireEvent.change(screen.getByRole("textbox", { name: "Tag" }), { target: { value: "cell-2" } });

    expect(screen.getByRole("textbox", { name: "Tag" })).toHaveValue("cell-2");
  });

  it("should record the typed RPM limit as a number, not a string", async () => {
    const user = userEvent.setup();
    const seen: TagRateLimitEntry[][] = [];
    render(<Harness initial={rowsWith("cell-1", null)} onValue={(v: TagRateLimitEntry[]) => seen.push(v)} />);

    fireEvent.change(screen.getByRole("spinbutton", { name: "RPM limit" }), { target: { value: "60" } });

    const latest = seen[seen.length - 1][0];
    expect(latest.rpm_limit).toBe(60);
    expect(typeof latest.rpm_limit).toBe("number");
  });

  it("should reset the RPM limit to null when the field is cleared", async () => {
    const user = userEvent.setup();
    const seen: TagRateLimitEntry[][] = [];
    render(<Harness initial={rowsWith("cell-1", 60)} onValue={(v: TagRateLimitEntry[]) => seen.push(v)} />);

    await user.clear(screen.getByRole("spinbutton", { name: "RPM limit" }));

    expect(seen[seen.length - 1][0].rpm_limit).toBeNull();
  });

  it("should remove only the clicked row", async () => {
    const user = userEvent.setup();
    const initial: TagRateLimitEntry[] = [
      { id: "r1", tag: "keep-me", rpm_limit: 10 },
      { id: "r2", tag: "delete-me", rpm_limit: 20 },
    ];
    render(<Harness initial={initial} />);

    await user.click(screen.getAllByRole("button", { name: "Remove tag limit" })[1]);

    const tags = screen.getAllByRole("textbox", { name: "Tag" });
    expect(tags).toHaveLength(1);
    expect(tags[0]).toHaveValue("keep-me");
  });

  it("should not submit the surrounding form when a row is removed", async () => {
    const user = userEvent.setup();
    let submitted = false;
    render(
      <form
        onSubmit={() => {
          submitted = true;
        }}
      >
        <Harness initial={rowsWith("cell-1", 10)} />
      </form>,
    );

    await user.click(screen.getByRole("button", { name: "Remove tag limit" }));

    expect(submitted).toBe(false);
    expect(screen.queryAllByRole("textbox", { name: "Tag" })).toHaveLength(0);
  });
});

describe("tagRowsToLimits", () => {
  it("should map named rows with numeric limits into the rpm map", () => {
    expect(tagRowsToLimits([{ id: "a", tag: "cell-1", rpm_limit: 60 }])).toEqual({ tag_rpm_limit: { "cell-1": 60 } });
  });

  it("should drop rows with a blank tag or a null limit", () => {
    const rows: TagRateLimitEntry[] = [
      { id: "a", tag: "  ", rpm_limit: 60 },
      { id: "b", tag: "cell-2", rpm_limit: null },
    ];
    expect(tagRowsToLimits(rows)).toEqual({ tag_rpm_limit: {} });
  });
});

describe("tagLimitsToRows", () => {
  it("should rebuild rows from a stored rpm map", () => {
    const rows = tagLimitsToRows({ "cell-1": 60 });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ tag: "cell-1", rpm_limit: 60 });
  });

  it("should ignore non-numeric entries", () => {
    expect(tagLimitsToRows({ "cell-1": "sixty" })).toEqual([]);
  });
});

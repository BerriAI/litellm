import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod/v4";
import { TeamMetadataField } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import { useZodForm } from "@/lib/forms/useZodForm";
import MetadataKeyValueFields, {
  MetadataPair,
  metadataObjectToPairs,
  metadataPairsSchema,
  metadataPairsToObject,
} from "./MetadataKeyValueFields";

describe("metadataObjectToPairs", () => {
  it("returns an empty list for null or undefined metadata", () => {
    expect(metadataObjectToPairs(null)).toEqual([]);
    expect(metadataObjectToPairs(undefined)).toEqual([]);
  });

  it("keeps plain string values as-is", () => {
    expect(metadataObjectToPairs({ department: "research" })).toEqual([{ key: "department", value: "research" }]);
  });

  it("serializes non-string values as JSON", () => {
    expect(
      metadataObjectToPairs({
        tier: 3,
        beta: true,
        config: { region: "us" },
        tags: ["a", "b"],
        empty: null,
      }),
    ).toEqual([
      { key: "tier", value: "3" },
      { key: "beta", value: "true" },
      { key: "config", value: '{"region":"us"}' },
      { key: "tags", value: '["a","b"]' },
      { key: "empty", value: "null" },
    ]);
  });

  it("quotes string values that would otherwise parse as JSON, so types round-trip", () => {
    expect(metadataObjectToPairs({ code: "42", flag: "true" })).toEqual([
      { key: "code", value: '"42"' },
      { key: "flag", value: '"true"' },
    ]);
  });

  it("filters out excluded keys", () => {
    expect(
      metadataObjectToPairs({ department: "research", logging: [{ callback_name: "langfuse" }] }, new Set(["logging"])),
    ).toEqual([{ key: "department", value: "research" }]);
  });
});

describe("metadataPairsToObject", () => {
  it("returns an empty object for undefined pairs", () => {
    expect(metadataPairsToObject(undefined)).toEqual({});
  });

  it("keeps plain text values as strings", () => {
    expect(metadataPairsToObject([{ key: "department", value: "research" }])).toEqual({ department: "research" });
  });

  it("parses JSON values into their typed form", () => {
    expect(
      metadataPairsToObject([
        { key: "tier", value: "3" },
        { key: "beta", value: "true" },
        { key: "config", value: '{"region":"us"}' },
        { key: "code", value: '"42"' },
      ]),
    ).toEqual({ tier: 3, beta: true, config: { region: "us" }, code: "42" });
  });

  it("skips rows without a key and defaults a missing value to an empty string", () => {
    expect(metadataPairsToObject([{ key: "", value: "orphan" }, undefined, { key: "kept" }])).toEqual({ kept: "" });
  });

  it("round-trips a mixed-type metadata object losslessly", () => {
    const metadata = {
      department: "research",
      code: "42",
      tier: 3,
      beta: true,
      config: { region: "us", replicas: 2 },
    };
    expect(metadataPairsToObject(metadataObjectToPairs(metadata))).toEqual(metadata);
  });
});

interface HarnessProps {
  onFinish: (values: { metadata?: MetadataPair[] }) => void;
  initialMetadata?: MetadataPair[];
  schemaFields?: TeamMetadataField[];
  schemaLoading?: boolean;
}

const harnessSchema = z.object({ metadata: metadataPairsSchema });

const Harness: React.FC<HarnessProps> = ({ onFinish, initialMetadata, schemaFields, schemaLoading }) => {
  const form = useZodForm(harnessSchema, { defaultValues: { metadata: initialMetadata ?? [] } });
  return (
    <form onSubmit={form.handleSubmit((values) => onFinish(values))}>
      <MetadataKeyValueFields
        control={form.control}
        getValues={form.getValues}
        name="metadata"
        schemaFields={schemaFields}
        schemaLoading={schemaLoading}
      />
      <button type="submit">Save</button>
    </form>
  );
};

describe("MetadataKeyValueFields", () => {
  it("renders one row per existing pair", () => {
    render(
      <Harness
        onFinish={vi.fn()}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "tier", value: "3" },
        ]}
      />,
    );

    const keyInputs = screen.getAllByPlaceholderText("Key");
    const valueInputs = screen.getAllByPlaceholderText("Value");
    expect(keyInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["department", "tier"]);
    expect(valueInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["research", "3"]);
  });

  it("adds a row and submits the entered pair", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} />);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    fireEvent.change(screen.getByPlaceholderText("Key"), { target: { value: "cost_center" } });
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "eng-1" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "cost_center", value: "eng-1" }] });
    });
  });

  it("removes a row when its remove icon is clicked", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(
      <Harness
        onFinish={onFinish}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "tier", value: "3" },
        ]}
      />,
    );

    await user.click(screen.getAllByLabelText("Remove key-value pair")[0]);
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "tier", value: "3" }] });
    });
  });

  it("blocks submission on duplicate keys", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(
      <Harness
        onFinish={onFinish}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "department", value: "sales" },
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getAllByText("Duplicate key").length).toBeGreaterThan(0);
    });
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("blocks submission when a row is missing its key", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} />);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "orphan" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByText("Missing key")).toBeInTheDocument();
    });
    expect(onFinish).not.toHaveBeenCalled();
  });
});

describe("MetadataKeyValueFields with a declared schema", () => {
  const schema: TeamMetadataField[] = [
    { key: "cost_center", label: "Cost Center" },
    { key: "app_name", label: "Application Name" },
  ];

  it("should prepopulate one ordinary editable pair row per declared key", async () => {
    render(<Harness onFinish={vi.fn()} schemaFields={schema} />);

    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value)).toEqual([
        "cost_center",
        "app_name",
      ]);
    });
    screen.getAllByPlaceholderText("Key").forEach((input) => expect(input).toBeEnabled());
    expect(screen.getAllByLabelText("Remove key-value pair")).toHaveLength(2);
  });

  it("should submit a prepopulated key with its typed value", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} schemaFields={[{ key: "cost_center", label: "Cost Center" }]} />);

    fireEvent.change(await screen.findByPlaceholderText("Value"), { target: { value: "CC-1001" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "cost_center", value: "CC-1001" }] });
    });
  });

  it("should not add a second row for keys already present in the form", async () => {
    render(
      <Harness onFinish={vi.fn()} schemaFields={schema} initialMetadata={[{ key: "cost_center", value: "CC-1001" }]} />,
    );

    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value)).toEqual([
        "cost_center",
        "app_name",
      ]);
    });
    expect(screen.getAllByPlaceholderText("Value").map((input) => (input as HTMLInputElement).value)).toEqual([
      "CC-1001",
      "",
    ]);
  });

  it("should let the user remove a prepopulated row", async () => {
    const user = userEvent.setup();
    render(<Harness onFinish={vi.fn()} schemaFields={schema} />);

    await screen.findAllByPlaceholderText("Key");
    await user.click(screen.getAllByLabelText("Remove key-value pair")[0]);

    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value)).toEqual([
        "app_name",
      ]);
    });
  });

  it("should show a skeleton instead of the editor while the schema is loading", () => {
    render(<Harness onFinish={vi.fn()} schemaLoading />);

    expect(screen.getByTestId("metadata-schema-skeleton")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /add key-value pair/i })).not.toBeInTheDocument();
  });

  it("should seed rows when the schema arrives after an initial loading state", async () => {
    const onFinish = vi.fn();
    const { rerender } = render(<Harness onFinish={onFinish} schemaLoading />);

    rerender(<Harness onFinish={onFinish} schemaFields={schema} schemaLoading={false} />);

    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value)).toEqual([
        "cost_center",
        "app_name",
      ]);
    });
  });
});

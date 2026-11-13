import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import VectorStorePermissions from "./VectorStorePermissions";

vi.mock("../networking");

const accessToken = "test-token";

describe("VectorStorePermissions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.vectorStoreListCall).mockResolvedValue({
      data: [
        { vector_store_id: "store-a", vector_store_name: "Parks knowledge base" },
        { vector_store_id: "store-b", vector_store_name: "HR policies" },
      ],
    });
  });

  it("lists stores that come only from access groups, counts them, and names the groups", async () => {
    render(
      <VectorStorePermissions
        vectorStores={[]}
        inheritedVectorStores={[{ id: "store-a", accessGroupNames: ["kb-readers", "support"] }]}
        accessToken={accessToken}
      />,
    );

    const chip = await screen.findByTitle("Granted via access groups kb-readers, support. Full ID: store-a");
    expect(chip).toHaveTextContent("Parks knowledge base (store-a) (access group)");
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.queryByText("No vector stores configured")).not.toBeInTheDocument();
    expect(networking.vectorStoreListCall).toHaveBeenCalledWith(accessToken);
  });

  it("lists a store once, as the team's own, when a group grants it too", async () => {
    render(
      <VectorStorePermissions
        vectorStores={["store-a"]}
        inheritedVectorStores={[
          { id: "store-a", accessGroupNames: ["kb-readers"] },
          { id: "store-b", accessGroupNames: ["kb-readers"] },
        ]}
        accessToken={accessToken}
      />,
    );

    expect(await screen.findByText("Parks knowledge base (store-a)")).toBeInTheDocument();
    expect(screen.getByTitle("Granted via access group kb-readers. Full ID: store-b")).toHaveTextContent(
      "HR policies (store-b) (access group)",
    );
    expect(screen.queryByTitle(/Full ID: store-a/)).not.toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("keeps the empty state and skips the store list when nothing is granted", () => {
    render(<VectorStorePermissions vectorStores={[]} accessToken={accessToken} />);

    expect(screen.getByText("No vector stores configured")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(networking.vectorStoreListCall).not.toHaveBeenCalled();
  });
});

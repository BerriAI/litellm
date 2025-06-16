import { cx } from "@/lib/cva.config";
import {
  CellContext,
  Column,
  ColumnDef,
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import {
  ChevronDownIcon,
  PlusIcon,
  SearchIcon,
  SettingsIcon,
  TablePropertiesIcon,
  Trash2Icon,
  UserRoundIcon,
  UsersRoundIcon,
} from "lucide-react";
import { VirtualKey, virtualKeys } from "./data";
import { CSSProperties, Fragment, useMemo } from "react";
import { CreateVirtualKeyDialog } from "./create-virtual-key-dialog";
import { useDialogStore } from "@ariakit/react";

const columnHelper = createColumnHelper<VirtualKey>();

function getPinStyles(column: Column<any>): CSSProperties {
  const isPinned = column.getIsPinned();
  if (isPinned === false) {
    return {};
  }

  return {
    position: "sticky",
    right: `${column.getAfter("right")}px`,
    width: column.getSize(),
    zIndex: 1,
    borderLeft: ".5px solid rgba(0,0,0,.15)",
  };
}

function TitleCell(props: { cellContext: CellContext<VirtualKey, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return (
    <div className={cx("flex flex-col gap-1 items-start tracking-tight")}>
      <span className="text-blue-600 text-[14px] bg-blue-50 px-1 py-0.5 rounded">
        {virtualKey.alias}
      </span>

      <span className="text-[13px] text-neutral-400">{virtualKey.id}</span>
    </div>
  );
}

function TeamCell(props: { cellContext: CellContext<VirtualKey, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  if (!virtualKey.team) return null;
  return (
    <div>
      <span
        className={cx(
          "text-blue-600 inline-flex text-[13px] tracking-tight bg-blue-50 px-1 py-0.5 rounded",
          "truncate",
        )}
      >
        {virtualKey.team.name}
      </span>
    </div>
  );
}

function UserCell(props: { cellContext: CellContext<VirtualKey, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return virtualKey.user ? (
    <div className={cx("flex flex-col gap-0.5 items-start tracking-tight")}>
      <span className="text-[14px] text-neutral-800">
        {virtualKey.user.name}
      </span>

      <span className="text-[13px] text-neutral-400">
        {virtualKey.user.email}
      </span>
    </div>
  ) : null;
}

function ExpiresCell(props: { cellContext: CellContext<VirtualKey, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return virtualKey.expires ? (
    <span className="text-[14px] text-neutral-800">{virtualKey.expires}</span>
  ) : null;
}

function SpendCell(props: { cellContext: CellContext<VirtualKey, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return (
    <span className="text-[14px] text-neutral-800">{virtualKey.spend}</span>
  );
}

function BudgetCell(props: { cellContext: CellContext<VirtualKey, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return virtualKey.budget ? (
    <span className="text-[14px] text-neutral-800">{virtualKey.budget}</span>
  ) : null;
}

function BudgetResetCell(props: {
  cellContext: CellContext<VirtualKey, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;

  return virtualKey.budgetReset ? (
    <span className="text-[14px] text-neutral-800">
      {virtualKey.budgetReset}
    </span>
  ) : null;
}

function CreatedAtCell(props: {
  cellContext: CellContext<VirtualKey, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;
  return (
    <span className="text-[14px] text-neutral-800">{virtualKey.createdAt}</span>
  );
}

function LastUsedCell(props: {
  cellContext: CellContext<VirtualKey, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;

  return virtualKey.lastUsed ? (
    <span className="text-[14px] text-neutral-800">{virtualKey.lastUsed}</span>
  ) : null;
}

function ActionsCell(_props: {
  cellContext: CellContext<VirtualKey, unknown>;
}) {
  // const virtualKey = props.cellContext.row.original;

  return (
    <div className="flex items-center gap-2">
      <button className="inline-flex items-center gap-1 h-[24px] px-2 rounded bg-neutral-100">
        <Trash2Icon className="size-3.5 text-neutral-600" />
        {/* <span className="text-[12px] text-neutral-800 tracking-tight">
          Delete
        </span> */}
      </button>

      <button className="inline-flex items-center gap-1 h-[24px] px-2 rounded bg-neutral-100">
        <SettingsIcon className="size-3.5 text-neutral-600" />
        {/* <span className="text-[12px] text-neutral-800 tracking-tight">
          Edit
        </span> */}
      </button>
    </div>
  );
}

const columns: ColumnDef<VirtualKey>[] = [
  columnHelper.display({
    id: "title",
    header: "Key Alias",
    cell: (cellContext) => <TitleCell cellContext={cellContext} />,
  }),
  columnHelper.display({
    id: "team",
    header: "Team Alias",
    cell: (cellContext) => <TeamCell cellContext={cellContext} />,
  }),
  columnHelper.display({
    id: "user",
    header: "User",
    cell: (cellContext) => <UserCell cellContext={cellContext} />,
  }),
  columnHelper.display({
    id: "expires",
    header: "Expires",
    cell: (cellContext) => <ExpiresCell cellContext={cellContext} />,
  }),
  columnHelper.display({
    id: "spend",
    header: "Spend (USD)",
    cell: (cellContext) => <SpendCell cellContext={cellContext} />,
    size: 150,
  }),
  columnHelper.display({
    id: "budget",
    header: "Budget (USD)",
    cell: (cellContext) => <BudgetCell cellContext={cellContext} />,
    size: 150,
  }),
  columnHelper.display({
    id: "budgetReset",
    header: "Budget Reset",
    cell: (cellContext) => <BudgetResetCell cellContext={cellContext} />,
  }),
  columnHelper.display({
    id: "createdAt",
    header: "Created At",
    cell: (cellContext) => <CreatedAtCell cellContext={cellContext} />,
  }),
  columnHelper.display({
    id: "lastUsed",
    header: "Last Used",
    cell: (cellContext) => <LastUsedCell cellContext={cellContext} />,
  }),
  columnHelper.display({
    id: "actions",
    header: "Actions",
    cell: (cellContext) => <ActionsCell cellContext={cellContext} />,
    size: 100,
  }),
];

export function VirtualKeyTable() {
  const data = useMemo(() => virtualKeys, []);
  const table = useReactTable({
    columns,
    data,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.id,
    defaultColumn: {
      size: 180,
    },
    initialState: {
      columnPinning: {
        right: ["actions"],
      },
    },
  });

  const createVirtualKeyDialogStore = useDialogStore({ defaultOpen: true });

  return (
    <Fragment>
      <CreateVirtualKeyDialog store={createVirtualKeyDialogStore} />

      <div className="flex flex-col min-h-0 gap-3">
        <div className="flex shrink-0 items-center justify-between gap-4 min-w-0">
          <div className="h-[48px] flex items-center gap-4 px-2 bg-neutral-50 min-w-0 rounded-lg">
            <div className="relative">
              <span
                className={cx(
                  "absolute left-0 inset-y-0  pointer-events-none",
                  "flex items-center justify-center pl-2.5",
                )}
              >
                <SearchIcon
                  strokeWidth={2.5}
                  className="size-3 text-neutral-400"
                />
              </span>

              <input
                className={cx(
                  "w-[400px] h-[34px] truncate rounded-md bg-white pl-[26px]",
                  "border-none",
                  "ring-[0.7px] ring-black/[0.08]",
                  "text-[13px] font-normal tracking-tight text-neutral-900",
                  "placeholder:text-neutral-400",
                )}
                placeholder="Search key name, key alias, key id"
              />
            </div>

            <div className="flex items-center gap-2">
              <div
                className={cx(
                  "h-[34px] bg-white px-2.5 rounded-md",
                  "flex items-center gap-1",
                  "ring-[0.7px] ring-black/[0.08]",
                  "text-[11px] font-medium tracking-tight",
                )}
              >
                <div className="flex items-center gap-1">
                  <UsersRoundIcon
                    strokeWidth={2.5}
                    className="size-3 text-neutral-400"
                  />

                  <span className="text-neutral-400">Team:</span>
                </div>

                <span className="text-neutral-800">All Teams</span>
              </div>

              <div
                className={cx(
                  "h-[34px] bg-white px-2.5 rounded-md",
                  "flex items-center gap-1",
                  "ring-[0.7px] ring-black/[0.08]",
                  "shadow-md shadow-black/[0.05]",
                  "text-[11px] font-medium tracking-tight",
                )}
              >
                <div className="flex items-center gap-1">
                  <UserRoundIcon
                    strokeWidth={2.5}
                    className="size-3 text-neutral-400"
                  />

                  <span className="text-neutral-400">User:</span>
                </div>

                <span className="text-blue-600 bg-blue-50 px-1 py-0.5 rounded">
                  Mike Carson
                </span>
              </div>
            </div>

            <span className="text-[12px] text-neutral-900 tracking-tight pr-2">
              Reset filters
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div
              className={cx(
                "h-[36px] bg-white px-3 rounded-md",
                "flex items-center gap-1",
                "text-[12px] font-medium tracking-tight",
                "bg-neutral-100",
              )}
            >
              <TablePropertiesIcon
                strokeWidth={2.5}
                className="size-3.5 text-neutral-400"
              />

              <span className="text-neutral-900">Columns</span>
            </div>

            <button
              onClick={createVirtualKeyDialogStore.show}
              type="submit"
              className={cx(
                "h-[36px] px-3 rounded-md",
                "flex items-center gap-1",
                "text-[12px] font-medium tracking-tight",
                "indigo-button-3d",
              )}
            >
              <PlusIcon strokeWidth={2.5} className="size-3.5 text-white" />
              <span className="text-white">Create New Key</span>
            </button>
          </div>
        </div>

        <div
          className={cx(
            "ring-[0.7px] ring-black/[0.08]",
            "rounded overflow-x-auto overflow-y-auto min-h-0 relative w-full isolate",
          )}
        >
          <table
            style={{ width: `max(100%, ${table.getCenterTotalSize()}px)` }}
            className="border-separate border-spacing-0"
          >
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      style={{
                        width: header.getSize(),
                        ...getPinStyles(header.column),
                        zIndex: 2,
                      }}
                      className={cx(
                        "text-start font-medium text-[11px] text-neutral-500 uppercase tracking-wider",
                        "px-3 py-2.5",
                        "border-r border-neutral-200 last:border-none",
                        "sticky top-0 bg-neutral-100",
                        header.column.getIsLastColumn("center") && "border-r-0",
                      )}
                    >
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>

            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="group">
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className={cx(
                        "px-3 py-2.5",
                        "border-b border-neutral-100 group-last:border-none bg-white",
                        // cell.column.getIsPinned()
                        //   ? "bg-white/80 backdrop-blur-sm"
                        //   : "bg-white",
                      )}
                      style={{ ...getPinStyles(cell.column) }}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="px-4 text-[14px] flex items-center justify-between ">
          <p className="text-neutral-700">1 — 15 of 30 results</p>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div
                className={cx(
                  "bg-white rounded-md",
                  "ring-[0.7px] ring-black/[0.12]",
                  "shadow-md shadow-black/[0.08]",
                  "relative h-[32px]",
                )}
              >
                <div
                  className={cx(
                    "pointer-events-none absolute left-0 inset-y-0 pl-3",
                    "flex items-center",
                    "text-neutral-700 text-[11px]",
                  )}
                >
                  <span>Page</span>
                </div>

                <select
                  style={{ backgroundImage: "none" }}
                  className={cx(
                    "!appearance-none border-none rounded-[inherit] shadow-none bg-transparent",
                    "h-full tracking-tight text-[11px] py-0 pl-[41px] pr-8 m-0 text-neutral-700",
                  )}
                >
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                </select>

                <span className="absolute inset-y-0 pointer-events-none right-0 pr-2 flex items-center">
                  <ChevronDownIcon size={16} className="text-neutral-500" />
                </span>
              </div>

              <span className="text-neutral-400 text-[12px] tracking-tight">
                of 2 pages
              </span>
            </div>

            <span className="size-[3px] shrink-0 rounded-full bg-neutral-600" />

            <div className="flex items-center gap-1">
              <button
                className={cx(
                  "border-none rounded-none shadow-none",
                  "h-[32px] px-2 items-center justify-center rounded",
                  "text-neutral-400 text-[12px] tracking-tight",
                  "hover:bg-neutral-100 transition-colors duration-100",
                )}
              >
                Prev
              </button>

              <button
                className={cx(
                  "border-none rounded-none shadow-none",
                  "h-[32px] px-2 items-center justify-center rounded",
                  "text-neutral-800 text-[12px] tracking-tight",
                  "hover:bg-neutral-100 transition-colors duration-100",
                )}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </div>
    </Fragment>
  );
}

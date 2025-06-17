"use client";

import { cx } from "@/lib/cva.config";
import {
  ChevronDownIcon,
  KeyRoundIcon,
  PlusIcon,
  SearchIcon,
  SettingsIcon,
  TablePropertiesIcon,
  Trash2Icon,
  UserRoundIcon,
  UsersRoundIcon,
} from "lucide-react";
import {
  CSSProperties,
  Fragment,
  startTransition,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { jwtDecode } from "jwt-decode";
import { matchSorter } from "match-sorter";
import {
  authContext,
  AuthContext,
  useAuthContext,
  useGlobalOverlaysContext,
} from "./contexts";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { keyListCall } from "@/components/networking";
import { useDialogStore } from "@ariakit/react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  CellContext,
  Column,
  ColumnDef,
  createColumnHelper,
} from "@tanstack/react-table";
import { CreateVirtualKeyDialog } from "./create-virtual-key-dialog";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { usePagination } from "./pagination";
import { GlobalOverlaysProvider } from "./global-overlays";

function getCookie(name: string) {
  const cookieValue = document.cookie
    .split("; ")
    .find((row) => row.startsWith(name + "="));
  return cookieValue ? cookieValue.split("=")[1] : null;
}

function PaginationButton({
  className,
  ...props
}: React.ComponentProps<"button">) {
  return (
    <button
      className={cx(
        "border-none rounded-none shadow-none disabled:opacity-50",
        "h-[32px] px-2 items-center justify-center rounded",
        "text-neutral-800 text-[12px] tracking-tight",
        "enabled:hover:bg-neutral-100 transition-colors duration-100",
        className,
      )}
      {...props}
    />
  );
}

const columnHelper = createColumnHelper<KeyResponse>();

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

function TitleCell(props: { cellContext: CellContext<KeyResponse, unknown> }) {
  const virtualKey = props.cellContext.row.original;
  const overlays = useGlobalOverlaysContext();

  return (
    <div
      className={cx("flex flex-col gap-1 items-start tracking-tight min-w-0")}
    >
      <button
        className="text-blue-600 text-[14px] truncate max-w-[320px] bg-blue-50 px-1 py-0.5 rounded"
        onClick={() => overlays.editVirtualKey({ virtualKey })}
      >
        {virtualKey.key_alias}
      </button>

      <span className="text-[13px] text-neutral-400 truncate max-w-[200px]">
        {virtualKey.token}
      </span>
    </div>
  );
}

function TeamCell(props: { cellContext: CellContext<KeyResponse, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  if (!virtualKey.team_alias) return null;
  return (
    <div>
      <span
        className={cx(
          "text-blue-600 inline-flex text-[13px] tracking-tight bg-blue-50 px-1 py-0.5 rounded",
          "truncate",
        )}
      >
        {virtualKey.team_alias}
      </span>
    </div>
  );
}

function UserCell(props: { cellContext: CellContext<KeyResponse, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return virtualKey.user_email ? (
    <div className={cx("flex flex-col gap-0.5 items-start tracking-tight")}>
      <span className="text-[14px] text-neutral-800">
        {virtualKey.user_email}
      </span>

      {/* <span className="text-[13px] text-neutral-400">
        {virtualKey.user.email}
      </span> */}
    </div>
  ) : null;
}

function ExpiresCell(props: {
  cellContext: CellContext<KeyResponse, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;

  return (
    <span className="text-[14px] text-neutral-800">
      {virtualKey.expires
        ? new Date(virtualKey.expires).toLocaleDateString()
        : "Never"}
    </span>
  );
}

function SpendCell(props: { cellContext: CellContext<KeyResponse, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return (
    <span className="text-[14px] text-neutral-800">{virtualKey.spend}</span>
  );
}

function BudgetCell(props: { cellContext: CellContext<KeyResponse, unknown> }) {
  const virtualKey = props.cellContext.row.original;

  return (
    <span className="text-[14px] text-neutral-800">
      {virtualKey.max_budget ? virtualKey.max_budget : "Unlimited"}
    </span>
  );
}

function BudgetResetCell(props: {
  cellContext: CellContext<KeyResponse, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;

  return (
    <span className="text-[14px] text-neutral-800">
      {virtualKey.budget_reset_at
        ? new Date(virtualKey.budget_reset_at).toLocaleDateString()
        : "Never"}
    </span>
  );
}

function CreatedAtCell(props: {
  cellContext: CellContext<KeyResponse, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;
  return (
    <span className="text-[14px] text-neutral-800">
      {new Date(virtualKey.created_at).toLocaleDateString()}
    </span>
  );
}

function LastUsedCell(props: {
  cellContext: CellContext<KeyResponse, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;

  return virtualKey.updated_at ? (
    <span className="text-[14px] text-neutral-800">
      {new Date(virtualKey.updated_at).toLocaleDateString()}
    </span>
  ) : null;
}

function ActionsCell(props: {
  cellContext: CellContext<KeyResponse, unknown>;
}) {
  const virtualKey = props.cellContext.row.original;
  const overlays = useGlobalOverlaysContext();

  return (
    <div className="flex items-center gap-2">
      <button
        className="inline-flex items-center gap-1 h-[24px] px-2 rounded bg-neutral-100"
        onClick={() => overlays.deleteVirtualKey({ virtualKey })}
      >
        <Trash2Icon className="size-3.5 text-neutral-600" />
      </button>

      <button
        className="inline-flex items-center gap-1 h-[24px] px-2 rounded bg-neutral-100"
        onClick={() => overlays.editVirtualKey({ virtualKey })}
      >
        <SettingsIcon className="size-3.5 text-neutral-600" />
      </button>
    </div>
  );
}

const columns: ColumnDef<KeyResponse>[] = [
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

function Content() {
  const authCtx = useAuthContext();
  const [searchTerm, setSearchTerm] = useState("");

  const keysQuery = useQuery<KeyResponse[]>({
    queryKey: ["keys", "token"],
    initialData: () => [],
    queryFn: () =>
      keyListCall(authCtx.key, null, null, null, null, null, 1, 100).then(
        (res) => res.keys,
      ),
  });

  const [scrollContainer, setScrollContainer] = useState<HTMLDivElement | null>(
    null,
  );

  const allKeys = keysQuery.data;
  const filteredKeys = useMemo(() => {
    const processedSearchTerm = searchTerm.trim();
    if (!processedSearchTerm) return allKeys;

    return matchSorter(allKeys, processedSearchTerm, {
      keys: [
        "key_alias",
        { key: "token", threshold: matchSorter.rankings.CONTAINS },
        { key: "key_name", threshold: matchSorter.rankings.CONTAINS },
      ],
      sorter: (rankedItems) => rankedItems,
    });
  }, [allKeys, searchTerm]);

  const paginationState = usePagination({
    total: filteredKeys.length,
    scrollContainer,
  });
  const { start, end } = paginationState;
  const data = useMemo(() => {
    return filteredKeys.slice(start - 1, end);
  }, [filteredKeys, start, end]);

  const table = useReactTable({
    columns,
    data,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.token,
    defaultColumn: {
      size: 180,
      maxSize: 200,
    },
    initialState: {
      columnPinning: {
        right: ["actions"],
      },
    },
  });

  const createVirtualKeyDialogStore = useDialogStore({
    setOpen(open) {
      // Refetch virtual keys whenever the dialog closes
      if (open === false) {
        keysQuery.refetch();
      }
    },
  });

  return (
    <Fragment>
      <CreateVirtualKeyDialog store={createVirtualKeyDialogStore} />

      <div className="p-8 h-full flex flex-col min-h-0">
        <div className="flex mb-8 shrink-0 flex-col gap-2">
          <div className="flex items-center gap-3">
            <div
              className={cx(
                "size-[48px] shrink-0 flex justify-center items-center bg-indigo-50 rounded-lg",
                "border border-indigo-100",
              )}
            >
              {keysQuery.isFetching ? (
                <UiLoadingSpinner className="size-4 text-indigo-500" />
              ) : (
                <KeyRoundIcon className="size-6 text-indigo-600" />
              )}
            </div>

            <h1 className="text-[32px] tracking-tighter text-neutral-900">
              Virtual Keys Management
            </h1>
          </div>

          <p className="max-w-[480px] text-[15px]/[1.6] text-neutral-500 tracking-tight">
            Manage and monitor your API keys with secure access control and
            real-time usage insights
          </p>
        </div>

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
                  onChange={(event) => {
                    startTransition(() => {
                      setSearchTerm(event.target.value);
                    });
                  }}
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
                    // "shadow-md shadow-black/[0.05]",
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

                  <span className="text-neutral-800">All Users</span>

                  {/* <span className="text-blue-600 bg-blue-50 px-1 py-0.5 rounded">
                    Mike Carson
                  </span> */}
                </div>
              </div>

              {/* <span className="text-[12px] text-neutral-900 tracking-tight pr-2">
                Reset filters
              </span> */}
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
            ref={setScrollContainer}
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
                          "px-3 py-2.5 truncate",
                          "border-r border-neutral-200 last:border-none",
                          "sticky top-0 bg-neutral-100",
                          header.column.getIsLastColumn("center") &&
                            "border-r-0",
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
                          "px-3 py-2.5 overflow-x-hidden",
                          "border-b border-neutral-100 group-last:border-none bg-white",
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

          <div className="px-4 text-[14px] flex items-center justify-between">
            <p className="text-neutral-700">
              {start} — {end} of {paginationState.total} results
            </p>

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
                    value={paginationState.page.toString()}
                    onChange={(event) =>
                      paginationState.setPage(+event.target.value)
                    }
                  >
                    {Array.from({ length: paginationState.pages }).map(
                      (_, i) => (
                        <option key={i + 1} value={i + 1}>
                          {i + 1}
                        </option>
                      ),
                    )}
                  </select>

                  <span className="absolute inset-y-0 pointer-events-none right-0 pr-2 flex items-center">
                    <ChevronDownIcon size={16} className="text-neutral-500" />
                  </span>
                </div>

                <span className="text-neutral-400 text-[12px] tracking-tight">
                  of {paginationState.pages}{" "}
                  {paginationState.pages === 1 ? "page" : "pages"}
                </span>
              </div>

              <span className="size-[3px] shrink-0 rounded-full bg-neutral-600" />

              <div className="flex items-center gap-1">
                <PaginationButton
                  disabled={paginationState.canGoBack === false}
                  onClick={paginationState.goBack}
                >
                  Prev
                </PaginationButton>

                <PaginationButton
                  disabled={paginationState.canGoForward === false}
                  onClick={paginationState.goForward}
                >
                  Next
                </PaginationButton>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Fragment>
  );
}

export default function VirtualKeysPage() {
  const [authContextValue, setAuthContextValue] = useState<AuthContext | null>(
    null,
  );
  const [queryClient] = useState(() => new QueryClient());

  // this is a temporary dirty hack and should not make it to prod
  useEffect(() => {
    const token = getCookie("token");
    setAuthContextValue(jwtDecode(token!) as AuthContext);
  }, []);

  if (authContextValue === null) return null;
  return (
    <QueryClientProvider client={queryClient}>
      <authContext.Provider value={authContextValue}>
        <GlobalOverlaysProvider>
          <Content />
        </GlobalOverlaysProvider>
      </authContext.Provider>
    </QueryClientProvider>
  );
}

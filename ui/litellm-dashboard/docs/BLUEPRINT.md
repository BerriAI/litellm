# Phase 1 migration blueprint

**Status:** 🔒 **Locked — later sections must follow these patterns; deviations
go in `DEVIATIONS.md`.**

Forged during Section 1 (Access Groups). Subsequent sections extend this file
only via the **blueprint stress test** in Section 2 (Virtual Keys); once
Section 2 commits, the blueprint is permanently locked.

---

## 1. Component translation table (antd → shadcn)

| antd component                   | shadcn replacement                                       | Import path                                                |
|----------------------------------|----------------------------------------------------------|------------------------------------------------------------|
| `Button`                         | `Button`                                                 | `@/components/ui/button`                                   |
| `Input`                          | `Input`                                                  | `@/components/ui/input`                                    |
| `Input.TextArea` / `TextArea`    | `Textarea`                                               | `@/components/ui/textarea`                                 |
| `Select` (single)                | `Select` + `SelectTrigger`/`SelectContent`/`SelectItem`  | `@/components/ui/select`                                   |
| `Select` (multi, `mode="multiple"`) | Custom `MultiSelect` (shadcn `Select` + chip list)    | see `AccessGroupBaseForm.tsx` for the canonical example    |
| `Checkbox`                       | `Checkbox`                                               | `@/components/ui/checkbox`                                 |
| `Switch`                         | `Switch`                                                 | `@/components/ui/switch`                                   |
| `Radio.Group`                    | `RadioGroup` + `RadioGroupItem`                          | `@/components/ui/radio-group`                              |
| `Form` + `Form.Item`             | `<form>` + `react-hook-form` + shadcn `Form`*            | `@/components/ui/form` (+ RHF `FormProvider` / `Controller` / `register`) |
| `Modal`                          | `Dialog` (`DialogContent`, `DialogHeader`, `DialogTitle`, `DialogDescription`, `DialogFooter`) | `@/components/ui/dialog` |
| `Drawer` / long-form modal       | `Sheet` (`SheetContent`, `SheetHeader`, `SheetTitle`)    | `@/components/ui/sheet`                                    |
| `Popover`                        | `Popover` + `PopoverTrigger` + `PopoverContent`          | `@/components/ui/popover`                                  |
| `Tooltip`                        | `Tooltip` + `TooltipTrigger` + `TooltipContent` inside a `TooltipProvider` | `@/components/ui/tooltip`                |
| `Dropdown` / `Dropdown.Button`   | `DropdownMenu` + `DropdownMenuTrigger` + `DropdownMenuContent` + `DropdownMenuItem` | `@/components/ui/dropdown-menu`    |
| `Menu` (application nav)         | `NavigationMenu` (top-level nav) **or** `DropdownMenu` (context menus) | `@/components/ui/navigation-menu` / `@/components/ui/dropdown-menu` |
| `Tabs` + `Tabs.TabPane`          | `Tabs` + `TabsList` + `TabsTrigger` + `TabsContent`      | `@/components/ui/tabs`                                     |
| `Tag`                            | `Badge` (`variant="secondary"` / `"outline"` / `"default"` / `"destructive"`) | `@/components/ui/badge`               |
| `Table` (basic)                  | `Table` + `TableHeader`/`TableBody`/`TableRow`/`TableHead`/`TableCell`, driven by `@tanstack/react-table` | `@/components/ui/table` |
| `Card`                           | `Card`                                                   | `@/components/ui/card`                                     |
| `Alert`                          | `Alert`                                                  | `@/components/ui/alert`                                    |
| `Skeleton` / `Spin`              | `Skeleton`                                               | `@/components/ui/skeleton`                                 |
| `Space`                          | Plain `<div className="flex gap-X items-center">`        | n/a — use Tailwind                                         |
| `Layout` / `Content` / `Sider`   | Plain `<div className="min-h-screen">` / `<main>` / `<aside>` + Tailwind | n/a — use Tailwind                         |
| `Divider`                        | `Separator`                                              | `@/components/ui/separator`                                |
| `Empty`                          | Custom empty-state div (see details-page example)        | inline                                                     |
| `Typography.Title`               | `<h1>` / `<h2>` / `<h3>` with Tailwind text-X classes    | n/a                                                        |
| `Typography.Text`                | `<span>` / `<p>` with Tailwind `text-muted-foreground`, etc. | n/a                                                    |
| `Typography.Text ellipsis`       | `<span className="truncate">` + `<Tooltip>` fallback     | `@/components/ui/tooltip`                                  |
| `Typography.Text copyable`       | Custom `CopyableId` (`navigator.clipboard.writeText` + `toast.success`) | inline (see details-page example)           |
| `Descriptions`                   | Plain `<dl>` / `<dt>` / `<dd>` with Tailwind             | n/a                                                        |
| `List.Item` grids                | Tailwind `grid grid-cols-{1..4}` + `Card` items          | `@/components/ui/card`                                     |
| `Pagination`                     | Custom prev/next buttons + page indicator (see list page) | inline (see `AccessGroupsPage.tsx`)                       |
| `Flex`                           | `<div className="flex ...">` with Tailwind classes       | n/a                                                        |
| `Row` / `Col`                    | Tailwind `grid grid-cols-X gap-Y`                        | n/a                                                        |
| `theme.useToken()`               | Tailwind semantic tokens (`bg-primary`, etc.)            | n/a                                                        |
| `ConfigProvider`                 | Not needed; shadcn CSS vars are configured in `globals.css` / `tailwind.config.ts` | n/a                                    |
| `message.*`                      | `MessageManager.*` (delegates to sonner internally)      | `@/components/molecules/message_manager`                   |
| `notification.*`                 | `NotificationManager.*` (delegates to sonner internally) | `@/components/molecules/notifications_manager`             |

### Notes

- **shadcn `Form` vs. manual RHF wiring.** For simple forms where we don't
  need `FormDescription` / `FormMessage` slots, direct `register(...)` +
  manual error text is fine (see `AccessGroupBaseForm.tsx`). For long forms
  with many validated inputs, use the full shadcn `<Form>` wrapper
  components. Both integrate the same RHF context, so there is no conflict
  when mixing.
- **Typography ellipsis.** Tailwind's `truncate` + wrapping `<Tooltip>` is
  the pattern we adopt. Don't reach for antd's `Typography.Text` for text
  truncation anymore.
- **`theme.useToken()` / `token.paddingLG`.** Replace with `p-6` / `px-12`
  (16 / 24 / 32px increments map cleanly to Tailwind's 4 / 6 / 8 / 12).

---

## 2. Icon translation table

| Source                                           | lucide-react equivalent    |
|--------------------------------------------------|----------------------------|
| `@ant-design/icons` → `PlusOutlined`             | `Plus`                     |
| `@ant-design/icons` → `SearchOutlined`           | `Search`                   |
| `@ant-design/icons` → `ExclamationCircleOutlined`| `AlertCircle`              |
| `@ant-design/icons` → `KeyOutlined`              | `Key`                      |
| `@ant-design/icons` → `BlockOutlined`            | `Box`                      |
| `@ant-design/icons` → `TeamOutlined`             | `Users`                    |
| `@ant-design/icons` → `UserOutlined`             | `User`                     |
| `@ant-design/icons` → `BankOutlined`             | `Building`                 |
| `@ant-design/icons` → `BookOutlined`             | `BookOpen`                 |
| `@ant-design/icons` → `BarChartOutlined`         | `BarChart3`                |
| `@ant-design/icons` → `LineChartOutlined`        | `LineChart`                |
| `@ant-design/icons` → `SettingOutlined`          | `Settings`                 |
| `@ant-design/icons` → `ToolOutlined`             | `Wrench`                   |
| `@ant-design/icons` → `SafetyOutlined`           | `Shield`                   |
| `@ant-design/icons` → `AuditOutlined`            | `ClipboardCheck`           |
| `@ant-design/icons` → `ApiOutlined`              | `Plug`                     |
| `@ant-design/icons` → `DatabaseOutlined`         | `Database`                 |
| `@ant-design/icons` → `TagsOutlined`             | `Tags`                     |
| `@ant-design/icons` → `CreditCardOutlined`       | `CreditCard`               |
| `@ant-design/icons` → `FileTextOutlined`         | `FileText`                 |
| `@ant-design/icons` → `FolderOutlined`           | `Folder`                   |
| `@ant-design/icons` → `AppstoreOutlined`         | `LayoutGrid`               |
| `@ant-design/icons` → `PlayCircleOutlined`       | `PlayCircle`               |
| `@ant-design/icons` → `RobotOutlined`            | `Bot`                      |
| `@ant-design/icons` → `BgColorsOutlined`         | `Palette`                  |
| `@ant-design/icons` → `ExperimentOutlined`       | `FlaskConical`             |
| `@ant-design/icons` → `ExportOutlined`           | `ExternalLink`             |
| `@ant-design/icons` → `ArrowLeftOutlined`        | `ArrowLeft`                |
| `@ant-design/icons` → `EditOutlined`             | `Pencil` or `Edit`         |
| `@ant-design/icons` → `DeleteOutlined`           | `Trash2`                   |
| `@ant-design/icons` → `CloseOutlined`            | `X`                        |
| `@ant-design/icons` → `CheckOutlined`            | `Check`                    |
| `@ant-design/icons` → `CopyOutlined`             | `Copy`                     |
| `@ant-design/icons` → `DownOutlined`             | `ChevronDown`              |
| `@ant-design/icons` → `UpOutlined`               | `ChevronUp`                |
| `@ant-design/icons` → `LoadingOutlined`          | `LoaderCircle` (add `animate-spin`) |
| `@heroicons/react/*`                             | lucide equivalent (same principle) |
| `@remixicon/react/*`                             | lucide equivalent          |

**When in doubt:** search <https://lucide.dev/icons> for the closest
name-match. lucide's icon set is a strict superset of what the ant /
heroicons usage in this repo needed as of phase 1.

---

## 3. Toast pattern

### Before (antd)

```tsx
import { message, notification } from "antd";

message.success("Created successfully");
notification.error({ message: "Error", description: err.message });
```

### After (sonner via global managers)

```tsx
import MessageManager from "@/components/molecules/message_manager";
import NotificationManager from "@/components/molecules/notifications_manager";

MessageManager.success("Created successfully");
NotificationManager.error("Something went wrong");
// Backend errors are best classified via fromBackend:
NotificationManager.fromBackend(err);
```

The managers delegate to sonner's `toast.*` under the hood. `<Toaster />`
is mounted once at the root layout. Do **not** render additional Toasters
from section code.

---

## 4. Form pattern (react-hook-form + zod + shadcn `<Form>`)

### Simple form — direct `register()`

Use when the form has ~5 or fewer fields and no complex async validation.
Suitable for the create / edit modals exercised in Section 1. See
`src/components/AccessGroups/AccessGroupsModal/AccessGroupBaseForm.tsx`.

```tsx
import { FormProvider, useForm, useFormContext, Controller } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type Values = { name: string; description: string; modelIds: string[] };

// Parent
function CreateModal() {
  const form = useForm<Values>({
    defaultValues: { name: "", description: "", modelIds: [] },
  });
  const onSubmit = form.handleSubmit((values) => { /* … */ });
  return (
    <FormProvider {...form}>
      <form onSubmit={onSubmit}>
        <FormFields />
        <Button type="submit">Save</Button>
      </form>
    </FormProvider>
  );
}

// Child — reads context
function FormFields() {
  const { register, control, formState } = useFormContext<Values>();
  return (
    <>
      <div className="space-y-2">
        <Label htmlFor="name">Name <span className="text-destructive">*</span></Label>
        <Input id="name" {...register("name", { required: "Please enter a name" })} />
        {formState.errors.name && (
          <p className="text-sm text-destructive">{formState.errors.name.message as string}</p>
        )}
      </div>
      <Controller
        control={control}
        name="modelIds"
        render={({ field }) => <MultiSelect value={field.value} onChange={field.onChange} /* … */ />}
      />
    </>
  );
}
```

### Complex form — shadcn `<Form>` components

Use when you need `FormDescription` / `FormMessage` slots and consistent
error rendering across many fields.

```tsx
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage,
} from "@/components/ui/form";

const schema = z.object({
  alias: z.string().min(1, "Alias required"),
  maxBudget: z.number().nonnegative().nullable(),
});
type Values = z.infer<typeof schema>;

const form = useForm<Values>({
  resolver: zodResolver(schema),
  defaultValues: { alias: "", maxBudget: null },
});

<Form {...form}>
  <form onSubmit={form.handleSubmit(onSubmit)}>
    <FormField
      control={form.control}
      name="alias"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Alias</FormLabel>
          <FormControl><Input {...field} /></FormControl>
          <FormDescription>Human-readable name for this key.</FormDescription>
          <FormMessage />
        </FormItem>
      )}
    />
  </form>
</Form>
```

### Schema location

- **Small schemas** (≤ 5 fields): inline in the form component.
- **Larger schemas**: colocated at `<form-component>.schemas.ts` next to the
  form file. Exported as `<name>Schema` and `<name>Values` type.

### Submit wiring

The submit handler continues to call the existing `useMutation` hook (from
the existing data layer). Don't replace the data-layer code in phase 1.

---

## 5. Table pattern

shadcn `<Table>` + `@tanstack/react-table`. See
`src/components/AccessGroups/AccessGroupsPage.tsx` for the canonical
example.

Skeleton:

```tsx
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  ColumnDef, flexRender, getCoreRowModel, getSortedRowModel,
  SortingState, useReactTable,
} from "@tanstack/react-table";

const columns = useMemo<ColumnDef<Row>[]>(() => [ /* … */ ], []);
const [sorting, setSorting] = useState<SortingState>([]);
const table = useReactTable<Row>({
  data: rows,
  columns,
  state: { sorting },
  onSortingChange: setSorting,
  getCoreRowModel: getCoreRowModel(),
  getSortedRowModel: getSortedRowModel(),
  getRowId: (r) => r.id,
});

<Table>
  <TableHeader>
    {table.getHeaderGroups().map((hg) => (
      <TableRow key={hg.id}>
        {hg.headers.map((h) => (
          <TableHead key={h.id}>
            {h.isPlaceholder ? null : flexRender(h.column.columnDef.header, h.getContext())}
          </TableHead>
        ))}
      </TableRow>
    ))}
  </TableHeader>
  <TableBody>
    {rows.length === 0 ? (
      <TableRow>
        <TableCell colSpan={columns.length} className="text-center text-muted-foreground py-8">
          No rows
        </TableCell>
      </TableRow>
    ) : rows.map((r) => (
      <TableRow key={r.id}>
        {r.getVisibleCells().map((c) => (
          <TableCell key={c.id}>{flexRender(c.column.columnDef.cell, c.getContext())}</TableCell>
        ))}
      </TableRow>
    ))}
  </TableBody>
</Table>
```

Column headers that support sort use the existing
`TableHeaderSortDropdown` (`src/components/common_components/TableHeaderSortDropdown/`).

**Pagination** is rendered separately (see `AccessGroupsPage.tsx`). Don't
use antd `<Pagination>` anymore — it's banned.

---

## 6. Shared layout patterns

### Page header

```tsx
<div className="flex justify-between items-center mb-4">
  <div>
    <h2 className="text-2xl font-semibold m-0">{title}</h2>
    <p className="text-muted-foreground text-sm m-0">{subtitle}</p>
  </div>
  <Button onClick={primaryAction}>Primary Action</Button>
</div>
```

### Section card

```tsx
<Card className="p-6">
  <h3 className="text-lg font-semibold mb-3">Section title</h3>
  {/* content */}
</Card>
```

### Empty state

```tsx
function EmptyState({ description }: { description: string }) {
  return (
    <div className="py-12 flex flex-col items-center justify-center text-muted-foreground">
      <div className="text-sm">{description}</div>
    </div>
  );
}
```

### Loading skeleton

```tsx
import { Skeleton } from "@/components/ui/skeleton";

<Skeleton className="h-10 w-full" />
<Skeleton className="h-6 w-32" />
```

### Error alert

```tsx
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

<Alert variant="destructive">
  <AlertTitle>Failed to load</AlertTitle>
  <AlertDescription>{error.message}</AlertDescription>
</Alert>
```

### Permissions gate

Keep the existing `useAuthorized` hook. Render-gate with early return or a
`PermissionsGate` wrapper — no new pattern introduced in phase 1.

---

## 7. Test-update checklist

When updating existing RTL tests for a migrated section:

1. **Mock `@/components/molecules/message_manager` / `notifications_manager`** —
   these modules now delegate to sonner but tests should still assert against
   the manager, not sonner directly (agnostic of backing library).
2. **Replace antd-specific selectors** (`.ant-btn`, `.ant-modal-title`, ...)
   with semantic queries (`getByRole`, `getByText`, `getByLabelText`).
3. **Global `setupTests.ts` mocks `notifications_manager`** by default. If a
   test needs to assert against the real manager, use `vi.unmock()`
   (see `notifications_manager.test.tsx`).
4. **JSDOM warnings about `PointerEvent` / `getPropertyValue`** are normal
   noise from Radix primitives and can be ignored; tests still pass.

---

## 8. Migration checklist per file

Apply this order for each file touched by the section:

1. Delete ALL `antd` / `@ant-design/icons` imports.
2. Delete ALL `@heroicons/react` / `@remixicon/react` imports.
3. Replace with shadcn primitives + lucide icons per the translation tables.
4. Replace raw color classes (`bg-slate-500`, `text-blue-600`, …) with
   semantic tokens (`bg-primary`, `text-foreground`, `bg-muted`,
   `border-border`, `text-destructive`, …).
5. Rewrite forms: `Form.useForm` → `useForm` from `react-hook-form`;
   `Form.Item` → direct `<Label>` + `<Input>` + `{...register}` or
   `Controller`. `form.validateFields()` → `form.handleSubmit(onSubmit)`.
6. If the file imports `antd.message` / `antd.notification` directly, route
   through `MessageManager` / `NotificationManager` instead.
7. Run `npx tsc --noEmit`, `npm run lint`, `npx vitest run <path>`. All
   must pass.

Files that are known to still import antd but are not in the current
section's scope are left untouched; the ESLint `no-banned-ui-imports` rule
reports them as expected.

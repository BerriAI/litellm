# Chat UI Design Reference

Tactical guide for building chat UI features. Copy patterns exactly — don't improvise spacing, colors, or components.

> **Agents:** Read `AGENTS.md` alongside this file. It contains decision trees, the pre-commit checklist, and the rules that govern when to use each pattern here. This file is the _what_; `AGENTS.md` is the _when and how_.

---

## Stack

- **Next.js 16** App Router, TypeScript
- **Tailwind CSS v4** — utility classes only, no custom CSS except in `globals.css`
- **shadcn/ui**: import from `@/components/ui/*`. Available today: `alert-dialog`, `badge`, `bubble`, `button`, `card`, `collapsible`, `dialog`, `input`, `input-group`, `label`, `popover`, `scroll-area`, `select`, `separator`, `sheet`, `skeleton`, `sonner`, `switch`, `table`, `tabs`, `textarea`, `tooltip`. Use these shared primitives when composing chat features
- **lucide-react** — only icon library, no emoji in UI
- System font stack (`-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`), inherited from `ChatShell`'s root

---

## Tokens

Never hardcode hex values. Use these CSS variables via Tailwind classes — all verified present in `src/app/globals.css`.

### Colors

| Token                         | Tailwind class                        | Use                                                        |
| ----------------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `--background`                | `bg-background`                       | Main content area, input surfaces                          |
| `--foreground`                | `text-foreground`                     | Primary text                                               |
| `--card`                      | `bg-card`                             | Card, popover and dialog surfaces                          |
| `--muted`                     | `bg-muted`                            | Table header rows, subtle fills                            |
| `--muted-foreground`          | `text-muted-foreground`               | Secondary/helper text, timestamps                          |
| `--border`                    | `border-border` (or bare `border`)    | All 1px separators                                         |
| `--primary`                   | `bg-primary` / `text-primary`         | Send button, checkmarks, active links                      |
| `--destructive`               | `bg-destructive` / `text-destructive` | Delete, error actions                                      |
| `--sidebar`                   | `bg-sidebar`                          | **Left sidebar background — use this, not `bg-secondary`** |
| `--sidebar-foreground`        | `text-sidebar-foreground`             | Sidebar text                                               |
| `--sidebar-accent`            | `bg-sidebar-accent`                   | Active/hover nav item fill                                 |
| `--sidebar-accent-foreground` | `text-sidebar-accent-foreground`      | Active nav item text                                       |
| `--sidebar-border`            | `border-sidebar-border`               | Sidebar's own dividers/right border                        |

**Known gotcha — verified in `globals.css`:** `--accent`, `--secondary`, and `--muted` all resolve to the _identical_ OKLCH value in both light and dark themes. Using `bg-accent` for a "selected" state against a `bg-secondary` container is **invisible** — there is zero contrast. This bit us repeatedly in this exact sidebar. Rules:

- Sidebar container: `bg-sidebar`, never `bg-secondary` or `bg-accent`.
- Active/selected nav item: `bg-sidebar-accent text-sidebar-accent-foreground`, never `bg-accent` on its own inside the sidebar.
- Anywhere else `bg-accent`/`bg-muted` is used for hover/selected state, confirm the container isn't _also_ `bg-accent`/`bg-secondary`/`bg-muted` before shipping — check `globals.css` values, don't assume.

### Status colors (semantic — don't substitute)

| State               | Class                                                    | Use                         |
| ------------------- | -------------------------------------------------------- | --------------------------- |
| Success / connected | `text-emerald-600 dark:text-emerald-400`                 | Completed, connected, ready |
| Running / warning   | `text-amber-600 dark:text-amber-400`                     | In-flight, expiring         |
| Error               | `text-red-600 dark:text-red-400` (or `text-destructive`) | Failed, expired             |
| Info                | `text-muted-foreground`                                  | Informational               |

---

## Typography

```tsx
// Page/greeting heading
<h1 className="text-[28px] font-semibold tracking-tight">Good afternoon</h1>

// Section heading (panel titles: "Your API Keys", "Your Usage")
<h2 className="text-base font-semibold tracking-tight">Your Usage</h2>

// Body text
<p className="text-sm text-foreground">Body</p>

// Secondary / helper text
<p className="text-xs text-muted-foreground">Helper</p>

// Table header
<th className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Col</th>

// Monospace (keys, IDs, JSON, tool args)
<span className="font-mono text-xs">sk-...abcd</span>
```

**Rules:**

- Use `tracking-tight` on headings
- Never set `text-foreground/50` or `text-foreground/70` for secondary text — use `text-muted-foreground`
- Monospace for: masked keys, conversation/tool-call IDs, tool arguments, JSON values

---

## Spacing

4px base grid.

| Context                            | Value                                       |
| ---------------------------------- | ------------------------------------------- |
| Sidebar outer padding              | `p-2` (8px)                                 |
| Nav item padding                   | `px-2.5 py-2`                               |
| Inline icon-to-label gap           | `gap-2.5` in nav items, `gap-1.5` elsewhere |
| Section divider margin             | `mx-2`                                      |
| Panel content padding              | `px-8 py-8`                                 |
| Card/bordered-div internal padding | `p-4`                                       |
| Dialog body padding                | `px-6 pb-6`                                 |

---

## Border Radius

```tsx
rounded-md // 8px — buttons, inputs, nav items, bordered divs
rounded-lg // 10px — panel cards, table containers
rounded-xl // 14px — chat input composer
rounded-2xl // 16px — chat message bubbles
rounded-full // pills, avatars, status dots
```

---

## Components

### Button

Always the shadcn `Button` — never a raw `<button>` with hand-rolled Tailwind, even for icon-only triggers. If it's borderline whether something is a "button" (e.g. a nav row, a suggestion pill), it still gets a `variant`.

```tsx
import { Button } from "@/components/ui/button"

<Button>Send</Button>
<Button variant="outline">Cancel</Button>
<Button variant="ghost">Nav item / low-emphasis</Button>
<Button variant="destructive">Delete</Button>
<Button variant="ghost" size="icon"><Settings className="size-4" /></Button>

// Loading
<Button disabled={loading}>
  {loading && <Loader2 className="size-4 animate-spin" />}
  Save
</Button>
```

Sizes: `default` (h-9), `sm` (h-8), `xs` (h-6), `lg` (h-10), `icon` (size-9), `icon-sm` (size-8), `icon-xs` (size-6).

### Sidebar nav item

No dedicated shadcn `Sidebar` block is installed (see gaps). Build nav items from `Button variant="ghost"`, full width, left-aligned, with the sidebar-specific active state:

```tsx
<Button
  variant="ghost"
  aria-current={active ? "page" : undefined}
  className={`w-full justify-start gap-2.5 px-2.5 font-medium hover:bg-sidebar-accent ${
    active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-muted-foreground"
  }`}
>
  <Icon className="size-4 shrink-0" />
  <span className="flex-1 text-left">{label}</span>
</Button>
```

There is no collapsible/icon-rail sidebar mode — it was removed (added complexity, no real use case, and its collapse toggle button had nowhere sensible to live). Don't reintroduce collapse state without a concrete reason.

### ScrollArea

**Known gotcha — cost real debugging time:** a `ScrollArea` won't actually scroll if its container is sized with `max-height` instead of `height`, even though the box _looks_ correctly capped. Radix's `ScrollArea` viewport is `size-full` (`height: 100%`), and CSS percentage-height resolution requires the ancestor chain to have a genuinely _definite_ height — `max-height` on a flex container doesn't count as definite for this purpose, even though the browser renders the box at the capped size. The result: the viewport silently grows to its full content height instead of 100% of the visible box, `scrollHeight === clientHeight` is reported (no overflow, so no scrollbar), and scroll events fall through to whatever's behind the popover/panel instead of scrolling the list.

```tsx
// Wrong — looks capped, doesn't scroll (max-height isn't a definite height)
<div className="w-[280px] max-h-[400px] flex flex-col overflow-hidden">
  <ScrollArea className="flex-1 min-h-0">{items}</ScrollArea>
</div>

// Right — explicit height propagates as definite through the whole chain
<div className="w-[280px] h-[400px] flex flex-col overflow-hidden">
  <ScrollArea className="flex-1 h-0">{items}</ScrollArea>
</div>
```

If you need "shrink to fit content, cap at N px" rather than "always N px," you cannot get both with pure CSS here — pick the fixed height (the common case: any list that can realistically exceed the cap, e.g. a model picker with hundreds of entries) over one that only works when short.

### Input / Textarea

Use `Input` and `Textarea` from `@/components/ui`. The shared `ChatComposer` uses `InputGroupTextarea` and owns submission, cancellation and IME handling. Reuse it for new chat surfaces

```tsx
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

<div className="flex flex-col gap-1.5">
  <Label htmlFor="alias">Key alias</Label>
  <Input id="alias" disabled />
  {error && <p className="text-xs text-destructive">{error}</p>}
</div>;
```

Errors go **below** the field. Never a toast for form validation.

### Card

Use the installed Card primitives for grouped content and action reviews

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

<Card size="sm">
  <CardHeader>
    <CardTitle>Review changes</CardTitle>
  </CardHeader>
  <CardContent>{children}</CardContent>
</Card>;
```

### Messages and Markdown

`ChatMessages` owns the shared list renderer. `ChatMessageContent` renders one message without a list wrapper so a transcript scroller can own each row. Both retain user editing, reasoning, tool output, copying and response metrics

Use the official registry `Bubble` and `BubbleContent` primitives. User messages use the muted variant and end alignment. Assistant messages use the ghost variant at full available width. Keep paragraph spacing and list markers in the shared Markdown renderer. GFM tables use the shared Table primitives, which provide horizontal scrolling. Forward Markdown cell styles so column alignment survives rendering

LiteAdmin passes `allowImages={false}` to show image descriptions without automatically requesting external image URLs from administrative answers. Other chat consumers retain image rendering by default

Scrolling belongs to the conversation container. LiteAdmin uses official MessageScroller items with stable chronological IDs; a pending action and its result stay in the same row. Its inline action review is an explicit product exception to the AlertDialog confirmation pattern below

### Badge

```tsx
import { Badge } from "@/components/ui/badge"

<Badge variant="secondary">connected</Badge>
<Badge variant="outline">expiring</Badge>
<Badge variant="destructive">expired</Badge>
```

### Dialog / AlertDialog

```tsx
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";

<Dialog open={open} onOpenChange={setOpen}>
  <DialogContent className="sm:max-w-lg">
    <DialogHeader>
      <DialogTitle>Rotate key</DialogTitle>
    </DialogHeader>
    <div className="flex flex-col gap-4">{/* form */}</div>
    <DialogFooter>
      <Button variant="outline" onClick={() => setOpen(false)}>
        Cancel
      </Button>
      <Button onClick={handleSubmit} disabled={loading}>
        {loading && <Loader2 className="size-4 animate-spin" />}
        Rotate
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>;
```

Use `AlertDialog` (not a manual confirm) for any destructive action — delete conversation, revoke credential.

### Table

```tsx
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";

<div className="border rounded-lg overflow-hidden">
  <Table>
    <TableHeader>
      <TableRow className="bg-muted/50">
        <TableHead className="text-[11px] font-medium uppercase tracking-wide">Key</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      <TableRow>
        <TableCell className="font-mono text-xs">sk-...abcd</TableCell>
      </TableRow>
    </TableBody>
  </Table>
</div>;
```

### Tabs

Always pass `variant="line"` for section tabs (Apps: All/Connected). The default variant renders a boxed pill with a background/shadow meant for compact toggle groups, not section tabs, and combining it with hand-added `border-b-2` classes produces a broken double-border look.

```tsx
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

<Tabs value={tab} onValueChange={setTab}>
  <TabsList variant="line" className="w-full justify-start border-b rounded-none h-auto p-0">
    <TabsTrigger value="all" className="rounded-none px-4 py-2 text-[13px]">
      All
    </TabsTrigger>
  </TabsList>
</Tabs>;
```

### Loading states

Skeleton loaders while fetching — never a bare spinner or blank screen for list/table content. `Loader2` + `animate-spin` is reserved for **inside a button** during a submit action, not as a page/panel loading state.

```tsx
import { Skeleton } from "@/components/ui/skeleton";

<div className="flex flex-col gap-3">
  <div className="border rounded-lg p-4 flex flex-col gap-2">
    <Skeleton className="h-4 w-1/3" />
    <Skeleton className="h-3 w-2/3" />
  </div>
</div>;
```

### Toast / notifications

**Resolved:** `sonner` is installed and mounted from the root layout. Raise toasts through `toast` from `@/lib/toast`; don't introduce a second toast mechanism.

---

## Page Layout Patterns

### Chat shell (sidebar + routed content)

```tsx
<div className="flex h-full w-full flex-col bg-background overflow-hidden">
  {/* optional full-width banner */}
  <div className="flex flex-1 min-h-0 overflow-hidden">
    <aside
      className="shrink-0 bg-sidebar border-r flex flex-col overflow-hidden"
      style={{ width: collapsed ? 56 : 260 }}
    >
      {/* nav items, conversation list */}
    </aside>
    <main className="flex-1 flex flex-col overflow-hidden min-w-0">{children}</main>
  </div>
</div>
```

### Routed panel (Integrations, Credentials, API Keys, Usage)

```tsx
<div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
  <PanelComponent />
</div>
```

No max-width cap on panel content — matches the rest of the dashboard's own pages (verified: `(dashboard)/api-keys/ApiKeysDashboard.tsx` has no width cap). A narrow `max-w-[800px]` here previously made wide tables (Keys) look cramped on real monitors; don't reintroduce it.

---

## Icons

`lucide-react` only. `size-4` (16px) inline/nav, `size-5` (20px) standalone actions, `size-3.5` (14px) inside badges/small buttons.

---

## Dos and Don'ts

| Do                                                                    | Don't                                                                               |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `Button` (any variant) for every clickable control                    | Raw `<button>` with hand-rolled Tailwind                                            |
| `bg-sidebar` / `bg-sidebar-accent` for sidebar surfaces               | `bg-secondary` / `bg-accent` for sidebar surfaces (identical values, zero contrast) |
| `Tabs variant="line"` for section tabs                                | Default `Tabs` variant + manual `border-b-2` overrides                              |
| Skeleton loaders for panel/list content                               | Spinner or blank screen while fetching                                              |
| `text-muted-foreground` for secondary text                            | `text-foreground/50`, `text-foreground/60`, `text-foreground/70`                    |
| `AlertDialog` for destructive confirmations                           | Instant delete, or a hand-rolled confirm                                            |
| Verify a prop/variant exists in `components/ui/*.tsx` before using it | Assume a shadcn prop exists because another shadcn app has it                       |
| `toast` from `@/lib/toast`, backed by the root sonner instance        | Introduce a second, competing toast library                                         |

---

## Known gaps (tracked, not blockers)

Card, Textarea and sonner are installed. Earlier versions of this document listed them as missing and encouraged parallel implementations; use the shared owners above

The shadcn Sidebar primitive block (`SidebarProvider`/`SidebarMenu`/etc) is still absent. Use the "Sidebar nav item" pattern above, built from `Button variant="ghost"` and the `sidebar-*` tokens

## File Structure

```
src/components/chat/
  ChatShell.tsx          # Sidebar chrome (nav, collapse, conversation list), shared across all /chat/* routes
  ConversationList.tsx   # Date-grouped chat list + Cmd+K search
  ChatMessages.tsx       # Shared list and single-message rendering (user, assistant, tool)
  MCPAppsPanel.tsx       # Integrations grid + detail view
  MCPConnectPicker.tsx   # MCP server toggle popover
  MCPCredentialsTab.tsx  # OAuth credentials table
  KeysPanel.tsx          # API key viewing + rotation
  UsagePanel.tsx         # Spend/usage analytics
  types.ts               # Shared TypeScript types
  useChatHistory.ts      # Chat persistence hook
  design.md              # This file
  AGENTS.md              # When/how to apply this file
../../contexts/ChatShellContext.tsx  # Cross-route state: MCP server selection, conversation history
../../app/chat/                      # Route tree: page.tsx (chats), layout.tsx (auth gate + shell),
                                      # integrations/, credentials/, api-keys/, usage/
```

# Chat UI Design Reference

Tactical guide for building chat UI features. Copy patterns exactly — don't improvise spacing, colors, or components.

The standalone `/ui/chat` workspace reuses Moyai's workspace structure and visual rules from `frontend/theme.css`, `native/web/style.css`, `brand.css`, `sidebar-layout.css`, and `home.css` at Moyai commit `1118ab0` (October 20, 2026). This is an intentional visual departure from the dashboard: neutral white/charcoal surfaces, restrained purple actions, and translucent navigation and composers. Shared controls and chat behavior remain owned by the existing LiteLLM primitives.

> **Agents:** Read `AGENTS.md` alongside this file. It contains decision trees, the pre-commit checklist, and the rules that govern when to use each pattern here. This file is the _what_; `AGENTS.md` is the _when and how_.

---

## Stack

- **Next.js 16** App Router, TypeScript
- **Tailwind CSS v4** — shared primitives and utilities, with the standalone Moyai port scoped in `globals.css` (theme and rail) and `moyai/moyai-chat.css` (home, composer and transcript)
- **shadcn/ui**: import from `@/components/ui/*`. Available today: `alert-dialog`, `badge`, `bubble`, `button`, `card`, `collapsible`, `dialog`, `input`, `input-group`, `label`, `popover`, `scroll-area`, `select`, `separator`, `sheet`, `skeleton`, `sonner`, `switch`, `table`, `tabs`, `textarea`, `tooltip`. Use these shared primitives when composing chat features
- **lucide-react** — only icon library, no emoji in UI
- Moyai's General Sans font from its original Fontshare stylesheet, with bundled Geist fallback; Geist Mono is bundled for model names

---

## Tokens

Never hardcode hex values. Use these CSS variables via Tailwind classes — all verified present in `src/app/globals.css`.

`ChatShell` applies `.chat-theme`, which scopes the palette to standalone Chat. Its dialogs and mobile Sheet also carry `.chat-theme` because they render through portals. Do not apply these overrides globally or to the Playground. Light and dark modes follow LiteLLM's existing theme switch. Keep status colors semantic.

### Glass surfaces

- `.chat-workspace`: static neutral backdrop, with no blur on this layout ancestor.
- `.chat-glass-sidebar`: Moyai's frosted navigation rail with its original neutral sheen.
- `.chat-glass-pane`: transparent on the home view so the source glass backdrop remains visible; quiet reading fill on conversation and workspace pages, without nested backdrop filtering.
- `.moyai-chat-home-composer`: Moyai's 10px rounded glass editor, 132px minimum desktop entry height, 14px blur, original sheen/shadow and bordered lower action row. `MoyaiChatComposer` owns its layout. Model selection sits at the right beside send. Message and code surfaces stay solid, using `--chat-message` / `--chat-message-border`; editor focus uses `--chat-focus-border`.
- `.chat-mobile-sidebar`: elevated, nearly opaque Sheet surface so navigation remains legible over conversation content.

Glass progressively enhances solid fills inside `@supports`. Reduced transparency and forced colors disable filtering, sheen and decorative shadows. Reduced motion disables chrome transitions. Use the original static dot/grid backdrop and source home heading measurements. Do not add animated backgrounds or an outer rounded frame.

The standalone `ChatModelPicker` exposes the account's chat and image-generation models from the composer action row. Its portalled popover carries `.chat-theme`; keep model names, capabilities and selection states readable in both themes. Attachment and MCP tools stay beside the composer.

The **Models** sidebar destination opens `/ui/chat/models` inside the same shell. It reuses the existing Model Hub catalog, filters, pricing and details in a read-only `chat` variant. Admins see the authenticated catalog and other users see the published catalog, matching the dashboard's existing access rules. The page uses `.moyai-models-page` for scrolling and `.chat-model-hub-card` for the bordered Moyai surface. Portalled model details carry `.chat-theme`; catalog management and other hub tabs stay in the dashboard.

Show available providers in a single-select dropdown beside model search. “All providers” clears the provider filter while keeping the other filters. The dropdown includes provider logos and uses the Chat theme in its portal. Provider logos reuse the existing asset mapping; white logo plates preserve the colors and contrast of the original assets in both themes. Unknown or failed logos fall back to a provider initial. The model list uses compact search and capability controls, quiet separators, and provider identities beside model names. Keep these styles scoped to Chat.

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

**Known gotcha:** generic `--accent`, `--secondary`, and `--muted` values may coincide in the dashboard theme. The standalone palette deliberately separates its sidebar selection from the chrome. Always use the sidebar-specific tokens rather than relying on incidental contrast from a generic token. Rules:

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
<header className="moyai-chat-home-header">
  <h1>What are we building?</h1>
</header> // Moyai: 500 weight, clamp(32px, 3.4vw, 46px), 1.13 line-height

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
| Sidebar outer padding              | `8px 10px 8px`                              |
| Nav item padding                   | `px-2.5 py-2`                               |
| Inline icon-to-label gap           | `gap-2.5` in nav items, `gap-1.5` elsewhere |
| Section divider margin             | `mx-2`                                      |
| Panel content padding              | `px-8 py-8`                                 |
| Card/bordered-div internal padding | `p-4`                                       |
| Dialog body padding                | `px-6 pb-6`                                 |

---

## Border Radius

```tsx
rounded - md; // 8px — buttons, inputs, nav items, bordered divs
rounded - lg; // 10px — panel cards, table containers
rounded - xl; // 12px — shared controls (Moyai composer uses its source 10px rule)
rounded - lg; // 8px — Moyai user message bubbles
rounded - full; // pills, avatars, status dots
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

The desktop sidebar follows Moyai: 240px wide, 230px below 1250px, and 52px when collapsed. Its top row uses the shared `ViewSwitcher`, search and collapse controls above a transparent 32px New Chat button and compact destinations. The switcher exposes AI Gateway, Moyai, Chat and registered plugins, with the active workspace marked. Chat uses the same rounded speech-bubble icon in the trigger and menu, including the dashboard entry point. The collapsed rail keeps the icon-only switcher above its expand button. Workspace keeps credentials, API keys, logs, usage and preview feedback reachable in one menu. Recent conversation rows show title, model and date at 48px height. The footer shows account identity, appearance, settings and sign out. The shared dashboard Navbar is absent from this standalone surface; only its product switcher is reused.

At 600px and below, a compact header keeps the workspace switcher beside the labelled menu button, visible even when the navigation Sheet is closed. The same navigation renders once inside the existing `Sheet`. Choosing a route or conversation closes the Sheet. Do not render two concurrent conversation lists: they each own a search shortcut listener. The collapsed desktop list remains mounted for keyboard search, but is visually hidden. The Chat layout owns an authenticated `PluginModeProvider` so returning to AI Gateway clears a previously selected plugin.

The landing page reuses Moyai's left-aligned 720px home content, dot/grid backdrop, 46px heading, 132px editor and compact starter rows. `ChatModelPicker` sits at the right of the composer action row, matching Moyai's model selector position. Selecting a model chooses chat or image generation. Preserve LiteLLM's existing model access and request behavior; do not copy Moyai product or backend logic.

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

Use `Input` and `Textarea` from `@/components/ui`. Standalone Chat uses `MoyaiChatComposer` for the source editor and toolbar arrangement, preserving submission, cancellation, attachments and IME handling. Uploads share a limit of four files per message and 10 MB per file. Images retain previews; PDF, DOCX, XLSX, text and code files use named file chips and native Responses `input_file` parts. File support depends on the selected model and provider. Attachment bytes stay in the existing user-scoped IndexedDB store; history metadata stays small. Legacy DOC/XLS files are not accepted. Other chat surfaces continue using the shared `ChatComposer` / `InputGroupTextarea`; this visual port is scoped to standalone Chat.

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
<div className="chat-theme chat-workspace flex h-full w-full flex-col overflow-hidden text-foreground">
  {/* optional full-width banner */}
  <div className="flex flex-1 min-h-0 overflow-hidden">
    <aside className="chat-glass-sidebar w-[248px] shrink-0 border-r border-sidebar-border flex flex-col overflow-hidden">
      {/* Desktop nav; on mobile use Sheet with the same content instead. */}
    </aside>
    <main className="chat-glass-pane flex-1 flex flex-col overflow-hidden min-w-0">{children}</main>
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
  ChatShell.tsx          # Desktop navigation / mobile Sheet, shared across all /chat/* routes
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

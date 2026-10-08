---
name: litellm-based-ui-design
description: Design or restyle a LiteLLM dashboard screen (ui/litellm-dashboard) in the house look the Usage page was rebuilt to, calm Braintrust and OpenAI-platform style on shadcn with LiteLLM's own colors. Carries the exact color scheme, chart palette, chart recipes, spacing and type scale, and component anatomy for page headers, stat strips with sparklines, the shared Model Leaderboard stacked chart, ranked lists with provider logos, segmented toggles, panels, dropdowns and dark mode, plus how to compare two or three variants live before committing to one. Use when someone asks to make a dashboard page look better, more modern, more polished, less ugly, more like Braintrust or OpenAI, consistent with Lens or Usage, or to restyle a tab, table, chart or card in the LiteLLM UI.
---

# LiteLLM based UI design

## The vibe

A dev tool that feels finished. Quiet neutral surfaces, one strong chart per screen, real numbers set large and readable, and color that only ever means data. It should feel like Braintrust's dashboards and the OpenAI platform usage page: calm, dense without being cramped, nothing decorative. It should never feel like an admin template (stacked cards with big bold titles, filled blue buttons for every toggle, rainbow badges, drop shadows everywhere)

Get close to those references without copying their layout or wording, and keep LiteLLM's own tokens and components. When in doubt, remove something

Three tests for any screen you touch

1. Squint at it. Only the chart and the headline numbers should carry visual weight. If a border, button or label competes with them, quiet it
2. Cover the color. The page should still read in grayscale, because structure comes from spacing and type, not from tinted boxes
3. Count the colors outside the charts. The answer is green and red on request counts and nothing else

## Reference implementation

The Usage page (`/usage`) is the canonical example. Read it before designing anything new and reuse its pieces rather than rebuilding them. Paths are under `ui/litellm-dashboard/src`

| File | What to take from it |
|---|---|
| `app/(dashboard)/usage/_components/components/overview/UsageOverview.tsx` | Whole-page composition: stat strip, chart card, ranking, keys |
| `.../overview/Primitives.tsx` | `Panel`, `Stat`, `Segmented`, `Sparkline`, `ChartSkeleton`, `LegendTotals` |
| `.../overview/BreakdownChart.tsx` | `Leaderboard` rows, `ModelMark` logo tile, `BreakdownControls` |
| `.../overview/overviewData.ts` | `seriesBy`, `bucketSeries`, `bucketTotals`, `dailyTotals`, `leaderboard`, number formatters |
| `.../overview/modelProvider.ts` | `providerForModel`, model name to provider logo |
| `components/shared/charts/StackedUsageChart.tsx` | The stacked chart and tooltip shared with the Model Leaderboard page |
| `app/(dashboard)/model-insights/_components/ModelInsightsView.tsx` | The Model Leaderboard page, the origin of the headline chart |

## Color scheme

Everything except chart fills comes from the theme tokens in `app/globals.css`, so light and dark both work for free. Never hardcode a gray

| Token | Light | Dark | Used for |
|---|---|---|---|
| `--background` | `oklch(1 0 0)` white | `oklch(0.248 0 0)` | Page, active segment pill, logo tiles |
| `--card` | `oklch(1 0 0)` white | `oklch(0.248 0 0)` | Every panel |
| `--muted` | `oklch(0.967 0.003 264.5)` cool off-white | `oklch(0.209 0 0)` | Segmented track, progress tracks, skeletons |
| `--muted-foreground` | `oklch(0.551 0.027 264.4)` slate | `oklch(0.754 0 0)` | Labels, subtitles, axis ticks, icons |
| `--foreground` | `oklch(0.13 0.028 261.7)` near-black navy | `oklch(0.964 0 0)` | Numbers, titles, names |
| `--border` | `oklch(0.928 0.006 264.5)` | `oklch(0.309 0 0)` | Panel borders, dividers, grid lines |
| `--success` | `oklch(0.527 0.154 150.1)` green | `oklch(0.792 0.209 151.7)` | The `N ok` count only |
| `--destructive` | `oklch(0.577 0.245 27.3)` red | `oklch(0.704 0.191 22.2)` | The `N failed` count and a nonzero Failed number |

The light theme is white panels on white with a cool slate tint in the grays, which is what makes it read as calm rather than flat. Do not add `bg-gray-50` page backgrounds or tinted card fills to "add depth"; the 1px border does that job

### Data colors

These are the only saturated colors on a screen, and they only ever appear inside charts, swatches and the dot on a logo tile

Brand blue `#2b3fd6` is LiteLLM's blue (the logo is `#0117be`; this is that hue lifted so a thin line and a 18% area fill both read). It is the color of every single-series chart: sparklines, latency, a lone bar series

The stacked palette, `STACKED_USAGE_PALETTE`, shared with the Model Leaderboard, is assigned in rank order so the biggest series always gets the same color

| Rank | Hex | Name |
|---|---|---|
| 1 | `#ec4899` | pink |
| 2 | `#a855f7` | purple |
| 3 | `#f59e0b` | amber |
| 4 | `#3b82f6` | blue |
| 5 | `#10b981` | emerald |
| 6 | `#ef4444` | red |
| 7 | `#14b8a6` | teal |
| 8 | `#84cc16` | lime |
| 9 | `#6366f1` | indigo |
| 10 | `#f97316` | orange |

The tail bucket `Other` is always slate `#94a3b8`, so it recedes behind the named series. Success and failure series in a chart are brand blue and `red`, not green and red, because green bars next to a pink leader look like a Christmas tree

Never use a gradient as a fill except the sparkline area (brand blue at 18% opacity fading to 0). Never put color on a panel, a header, a tab or a button

## Type and spacing

The dashboard font everywhere. Monospace only for raw IDs (key hashes), never for numbers, because `tabular-nums` on the normal font already aligns digits

| Role | Classes |
|---|---|
| Page title | `text-sm font-semibold` with the page's lucide icon at `size-4 strokeWidth={2}` |
| Panel title | `text-sm font-medium text-foreground`, optional lucide icon `size-4 strokeWidth={1.75} text-muted-foreground` |
| Panel subtitle | `text-xs text-muted-foreground`, one line, says what the chart is in plain words ("Daily spend, top 8 stacked") |
| Stat label | `text-xs text-muted-foreground` |
| Headline number | `text-xl` in a strip, `text-3xl` when it is the only number, always `font-semibold tracking-tight tabular-nums` |
| Context line under a number | `text-xs text-muted-foreground`, one line |
| Table and list text | `text-sm` names, `text-xs text-muted-foreground` secondary lines, numbers right-aligned with `tabular-nums` |

Spacing is a short ladder: `gap-3` between panels, `px-4 pt-3.5 pb-3` inside a stat cell, `px-5 pt-4` for a chart card header, `gap-2` between toolbar controls, `py-2.5` per list row. Corners are `rounded-xl` for panels, `rounded-lg` for the segmented track and skeletons, `rounded-md` for segments, tiles and menu items

## Page anatomy

Read top to bottom, the Usage page is the template for any data screen

1. Header row. `flex flex-wrap items-center justify-between gap-3 border-b pb-3 mb-3`. Left: page icon + title, then the view selector as a compact `h-8` select. Right: filters, then the date range. One row, never a stack of label-above-control blocks
2. Tab row. `TabsList variant="line"` (underline tabs, no pill background) on the left, secondary actions on the right as `Button variant="outline" size="sm"` with a lucide icon (Ask AI, Export Data)
3. Stat strip. One bordered panel split into 3 or 4 cells with `lg:divide-x`, not separate cards
4. Headline chart card. The stacked chart with its toggles in the header and the ranking under a `border-t` in the same card
5. Supporting panels. Tables and secondary charts, each a `Panel`, `gap-3` apart

Pages are full width with `px-4 sm:px-6 pt-3 pb-8`. Do not center a narrow column

## Components

### Panel

`rounded-xl border bg-card`, header `min-h-11 px-4 pt-3` with the title on the left and an optional action on the right, body `px-4 pt-2 pb-4`. No `shadow-*`, no colored header strip, no title larger than `text-sm`

### Stat strip

Each cell: label, number, one context line, then a full-width brand-blue `Sparkline` (`h-14`, stroke 1.5, area 18% fading to 0, monotone, no axes, no dots, no tooltip). Every cell gets a context line so the numbers line up; pick one that earns its place

- Spend: `of $X budget`, or `No budget set`
- Requests: `1,244 ok` in `text-success` and `107 failed` in `text-destructive`, plain words after the numbers
- Tokens: `8.6M cache read`
- The last cell can stack two small stats instead of a sparkline (avg cost per request over avg latency with the success rate as its context line)

### Headline chart

`StackedUsageChart`, 380px tall, `barCategoryGap="15%"`, `maxBarSize={64}`, horizontal grid only, no axis lines, no tick lines, x ticks at `minTickGap={48}`, y ticks formatted in the metric's unit. The tooltip is the default shadcn `ChartTooltipContent` with the header `Oct 3 · Total $64.21` and one row per series with its swatch. Do not build a custom tooltip

Top 8 series by the selected metric, the rest folded into slate `Other` so each bar still sums to the day's total. Header toggles, left to right: metric (Spend, Tokens, Requests), bucket (Daily, Weekly), scale (Linear, Log). Spend is the default metric because spend is what people open a usage page to see. Build the series on the client from the daily breakdown the page already loads so it follows the date range and filters without a new endpoint

### Ranked list

Same card as the chart, under `border-t px-5 py-3`, two columns on desktop with `divide-y divide-border/60` rows. Each row is a 4-column grid: rank in muted `text-xs`, `ModelMark` plus name with `by <provider>` under it, share percent with the absolute value under it, and the change in share between the two halves of the range as a muted arrow and number. No column header row; the values explain themselves

### ModelMark

A `size-7 rounded-md border bg-background` tile with the provider logo at `size-4` inside and a `size-2` dot in the series color on the bottom-right corner with `ring-2 ring-card`. It ties a list row to its chart color without coloring the row. `providerForModel` reads a `provider/` prefix first, then known families (`gpt-`, `o1` to OpenAI, `claude` to Anthropic, `gemini` to Google, and so on), and anything unknown renders a muted first-letter monogram, never a broken image. Rows whose key is already a provider use `ProviderLogo` directly

### Segmented

`inline-flex rounded-lg bg-muted p-0.5`, segments `rounded-md px-2.5 py-1 text-xs font-medium`, active `bg-background text-foreground shadow-xs`, inactive `text-muted-foreground hover:text-foreground`. `role="radiogroup"` with `role="radio"` segments. This replaces every filled toggle button, antd `Radio.Button` group and tremor `TabGroup` used as a view switch

### Selects and dropdowns

Triggers are `h-8`. A shadcn `SelectContent` defaults to the trigger's width, which clips any item with a description line, so pass `align="start" className="w-auto min-w-72 p-1"`. Items are `rounded-md py-2 pl-2 pr-8` with a muted icon, the label at `text-sm leading-5`, and the description at `text-xs leading-4 text-muted-foreground`

### Tables

Quiet header (`text-xs text-muted-foreground`, no fill, no uppercase), `text-sm` rows, numbers right-aligned `tabular-nums`, money with two decimals. Links to a detail view are the row's name, not a separate button

### Loading and empty

Skeletons match the shape they replace (`Skeleton` at the number's height and width, `ChartSkeleton` at the chart's height). Never show a previous range's numbers while a new range loads. Empty states are one muted line centered in the space the content would take, "No usage in this range", with no illustration

## Charts in general

Bars for counts over time, especially over few days; a smoothed line over five points invents hills the data does not have. Lines only for continuous measures like latency, and only with enough points. Area fills only on sparklines

Labels say what a person would say: `Successful` and `Failed`, never `successful_requests`. Money ticks use compact dollars (`$20`, `$1.2K`), token and request ticks use compact counts (`41.7M`)

## Dark mode

All chrome follows the tokens above. Chart fills stay the same hex in both themes. Check in dark before shipping: the stat strip borders, chart grid, tooltip surface, logo tiles (logos with dark ink rely on the existing `logoTreatmentFor` treatment) and the open dropdown

## How to run the work

Restyle, do not rewrite. Keep every fetch, prop, callback, `data-testid` and accessible name, and change presentation only. A redesign that also moves data logic is two PRs

Show before you commit. When the user wants a new look, build two or three variants behind a `?design=a|b|c` search param with a small fixed switcher pill, leave the current page as the no-param default, and run the dev server from the worktree so they can flip between them on real data. Once they pick one, make it the only path and delete the others and the switcher in the same PR. Users choose faster from a live page than from a description, and they usually pick the variant that keeps the existing layout and only polishes it

Look at it yourself. After every meaningful change take a screenshot at desktop width, read it, and fix what is off before reporting. A green typecheck says nothing about clipping, alignment or a broken logo

Capture the before screenshots from an untouched copy of the base branch (`git archive HEAD ui/litellm-dashboard` into a temp dir, copy `node_modules` in since Turbopack rejects a symlink that leaves the project, run it on another port), not from your working tree after you have started

## Shipping

Run, from `ui/litellm-dashboard` on the Node version in `.nvmrc`: `tsc --noEmit -p tsconfig.production.json`, `eslint` on the touched paths (the repo enforces PascalCase component filenames, no nested ternaries and the z-index scale from `globals.css`), `prettier --check`, `vitest` on the touched areas, and `knip` to delete exports the redesign left unused

Tests that asserted the old look get rewritten to assert the new markup with the same behavioral guarantee. Never delete a test to make a redesign pass, and keep the numbers a test proves on screen

The PR carries before and after screenshots of every view it touched, light and dark

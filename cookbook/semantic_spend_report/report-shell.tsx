import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync, writeFileSync } from "node:fs";
import {
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenuButton,
} from "@/components/shared/Sidebar";
import {
  PageHeader,
  PageHeaderTitle,
  PageHeaderDescription,
} from "@/components/shared/PageHeader";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Table, TableHeader, TableBody } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart3,
  Download,
  KeyRound,
  Box,
  List,
  ExternalLink,
} from "lucide-react";

const root = process.env.REPORT_ROOT!;
const logo = `data:image/png;base64,${readFileSync(`${process.env.LITELLM_REPO}/litellm/proxy/logo.png`).toString("base64")}`;
const gateway = "https://gateway.litellm-sandbox.ai";
const metrics = [
  [
    "Selected task spend",
    "total-spend",
    "Gateway-recorded costs",
    "spend-caption",
  ],
  ["Active sessions", "session-count", "In the selected view", ""],
  ["Model requests", "request-count", "Counted once per request", ""],
  ["Report run overhead", "overhead-spend", "", "overhead-caption"],
];

function Report() {
  return (
    <>
      <div className="fixed inset-y-0 left-0 hidden md:flex">
        <Sidebar>
          <SidebarHeader className="h-14 border-b border-border">
            <div className="flex items-center justify-between gap-2">
              <img
                src={logo}
                alt="LiteLLM"
                className="h-5 max-w-[150px] object-contain"
              />
              <Badge variant="outline">POC</Badge>
            </div>
          </SidebarHeader>
          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>AI Gateway</SidebarGroupLabel>
              <SidebarMenuButton
                render={
                  <a
                    href={`${gateway}/ui/?page=api-keys`}
                    target="_blank"
                    rel="noopener"
                  />
                }
              >
                <KeyRound />
                Virtual Keys
              </SidebarMenuButton>
              <SidebarMenuButton
                render={
                  <a
                    href={`${gateway}/ui/?page=models`}
                    target="_blank"
                    rel="noopener"
                  />
                }
              >
                <Box />
                Models
              </SidebarMenuButton>
            </SidebarGroup>
            <SidebarGroup>
              <SidebarGroupLabel>Observability</SidebarGroupLabel>
              <SidebarMenuButton
                render={
                  <a
                    href={`${gateway}/ui/usage`}
                    target="_blank"
                    rel="noopener"
                  />
                }
              >
                <BarChart3 />
                Usage
              </SidebarMenuButton>
              <SidebarMenuButton
                isActive
                render={<a href="#overview" aria-current="page" />}
              >
                <BarChart3 />
                Semantic Spend
              </SidebarMenuButton>
              <SidebarMenuButton
                render={
                  <a
                    href={`${gateway}/ui/?page=logs`}
                    target="_blank"
                    rel="noopener"
                  />
                }
              >
                <List />
                Logs
              </SidebarMenuButton>
            </SidebarGroup>
            <SidebarGroup>
              <SidebarGroupLabel>Report</SidebarGroupLabel>
              <SidebarMenuButton
                render={<a href="request-lineage.csv" download />}
              >
                <Download />
                Request export
              </SidebarMenuButton>
            </SidebarGroup>
          </SidebarContent>
          <SidebarFooter>
            <div className="text-sm font-medium">Sandbox report</div>
            <div className="text-xs text-muted-foreground">
              Synthetic demo data
            </div>
          </SidebarFooter>
        </Sidebar>
      </div>
      <div className="md:ml-[280px]">
        <header className="flex h-14 items-center justify-between border-b border-border px-4 text-sm">
          <div className="flex items-center gap-3">
            <span className="text-muted-foreground">AI Gateway</span>
            <span className="text-muted-foreground">›</span>
            <span>Semantic Spend</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            render={<a href={gateway} target="_blank" rel="noopener" />}
          >
            Open gateway
            <ExternalLink />
          </Button>
        </header>
        <main id="overview" className="px-4 pt-3 pb-8 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-4">
            <PageHeader>
              <PageHeaderTitle>
                <BarChart3 />
                Semantic Spend
              </PageHeaderTitle>
              <PageHeaderDescription>
                Where your AI spend goes, and the work behind it
              </PageHeaderDescription>
            </PageHeader>
            <Button
              variant="outline"
              size="sm"
              render={<a href="category-spend.csv" download />}
            >
              <Download />
              Export full-run CSV
            </Button>
          </div>
          <section className="filterbar" aria-label="Report filters">
            <label>
              User
              <select id="user-filter" aria-label="Filter user" />
            </label>
            <label>
              Reporting window
              <select id="period-filter">
                <option value="all">Full demo</option>
                <option value="first">First half</option>
                <option value="last">Second half</option>
                <option value="custom">Custom range</option>
              </select>
            </label>
            <label>
              From · UTC
              <Input id="start-filter" type="datetime-local" step="1" />
            </label>
            <label>
              To · UTC
              <Input id="end-filter" type="datetime-local" step="1" />
            </label>
            <Button
              variant="ghost"
              size="sm"
              id="reset-filters"
              className="self-end"
            >
              Reset
            </Button>
          </section>
          <p id="filter-error" className="error" role="alert" hidden />
          <div className="window-note">
            <span className="status-dot" />
            <span id="window-caption" />
            <span className="synthetic-label">
              Synthetic conversations · no measured ROI
            </span>
          </div>
          <section
            className="grid grid-cols-2 gap-3 mb-4 xl:grid-cols-4"
            aria-label="Selected metrics"
          >
            {metrics.map(([label, id, hint, hintId]) => (
              <Card size="sm" key={id}>
                <CardContent>
                  <div className="text-xs text-muted-foreground">{label}</div>
                  <strong
                    id={id}
                    className="block mt-1 text-2xl font-semibold tracking-tight tabular-nums"
                  />
                  <small
                    id={hintId || undefined}
                    className="text-xs text-muted-foreground"
                  >
                    {hint}
                  </small>
                </CardContent>
              </Card>
            ))}
          </section>
          <div className="charts">
            <Card size="sm">
              <CardHeader>
                <CardTitle>Session spend over time</CardTitle>
                <CardDescription>
                  Spend per bucket, stacked by session
                </CardDescription>
                <Badge variant="secondary" id="bucket-label" />
              </CardHeader>
              <CardContent>
                <div id="timeline" aria-label="Session spend over time" />
                <div id="session-legend" className="legend" />
              </CardContent>
            </Card>
            <Card size="sm">
              <CardHeader>
                <CardTitle>Spend share</CardTitle>
                <CardDescription>
                  All work areas in the selected window
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  id="share-bar"
                  className="share-bar"
                  aria-label="Category spend share"
                />
                <div id="share-legend" />
              </CardContent>
            </Card>
          </div>
          <Card size="sm" id="rankings">
            <CardContent>
              <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                <Tabs defaultValue="sessions">
                  <TabsList variant="line" aria-label="Ranking view">
                    <TabsTrigger
                      value="sessions"
                      data-view="sessions"
                      aria-pressed="true"
                    >
                      Sessions
                    </TabsTrigger>
                    <TabsTrigger
                      value="categories"
                      data-view="categories"
                      aria-pressed="false"
                    >
                      Work areas
                    </TabsTrigger>
                  </TabsList>
                </Tabs>
                <label className="text-xs text-muted-foreground">
                  Rank by{" "}
                  <select id="sort-filter">
                    <option value="spend">Spend</option>
                    <option value="requests">Requests</option>
                  </select>
                </label>
              </div>
              <div
                className="category-tabs"
                id="category-tabs"
                aria-label="Filter by work area"
              />
              <Table>
                <TableHeader id="ranking-head" />
                <TableBody id="ranking-body" />
              </Table>
              <div className="table-footer">
                <span id="ranking-caption" />
                <span>Click a name to explore findings and quotes</span>
              </div>
            </CardContent>
          </Card>
          <footer>
            <span>
              Recorded spend, not a productivity score. Unknown and mixed work
              stay visible.
            </span>
            <a href="verification.json">Accounting evidence ↗</a>
          </footer>
        </main>
      </div>
      <dialog id="detail-drawer" aria-labelledby="drawer-title">
        <div className="drawer-header">
          <div>
            <span id="drawer-eyebrow" className="eyebrow" />
            <h2 id="drawer-title" />
          </div>
          <Button
            variant="ghost"
            size="icon"
            id="close-drawer"
            aria-label="Close details"
          >
            ✕
          </Button>
        </div>
        <div id="drawer-content" className="drawer-content" />
      </dialog>
      <div id="chart-tooltip" className="chart-tooltip" role="tooltip" hidden />
      <noscript>
        <p>
          This interactive report needs JavaScript. CSV exports are available
          alongside this file.
        </p>
      </noscript>
    </>
  );
}

writeFileSync(
  `${root}/artifacts/report-shell.html`,
  renderToStaticMarkup(<Report />),
);

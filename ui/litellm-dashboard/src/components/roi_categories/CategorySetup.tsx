"use client";

import { useState } from "react";
import { Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { roiRequest } from "./api";
import type { Category, ContextFile, Settings } from "./data";

export function CategorySetup({
  token,
  settings,
  onSaved,
}: {
  token: string;
  settings: Settings | null;
  onSaved: () => void;
}) {
  const [sampleSize, setSampleSize] = useState(100);
  const [discoveryModel, setDiscoveryModel] = useState("");
  const [classifierModel, setClassifierModel] = useState("");
  const [teamId, setTeamId] = useState("");
  const [files, setFiles] = useState<ContextFile[]>([]);
  const [categories, setCategories] = useState<Category[]>(settings?.categories ?? []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function addFiles(incoming: FileList | null) {
    if (!incoming) return;
    setBusy(true);
    setError("");
    try {
      const extracted: ContextFile[] = [];
      for (const file of Array.from(incoming)) {
        if (file.size > 1024 * 1024) throw new Error("Each file must be at most 1 MiB");
        const content_base64 = await fileContent(file);
        extracted.push(await roiRequest<ContextFile>(token, "/context", { name: file.name, content_base64 }));
      }
      const combined = [...files, ...extracted];
      if (combined.map((file) => file.text).join("\n\n").length > 32000)
        throw new Error("Combined business context must fit within 32,000 characters");
      setFiles(combined);
    } catch (error) {
      setError(error instanceof Error ? error.message : "File extraction failed");
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    setBusy(true);
    setError("");
    try {
      if (settings?.phase === "error") await roiRequest(token, "/retry", {});
      else if (settings?.phase === "review") await roiRequest(token, "/activate", { categories });
      else {
        const setup = {
          sample_size: sampleSize,
          discovery_model: discoveryModel,
          classifier_model: classifierModel,
          team_id: teamId,
          context: files.map((file) => file.text).join("\n\n"),
        };
        await roiRequest(token, "/setup", setup);
      }
      onSaved();
    } catch (error) {
      setError(error instanceof Error ? error.message : "Unable to save setup");
    } finally {
      setBusy(false);
    }
  }

  if (settings?.phase === "sampling")
    return (
      <div className="rounded-lg border p-6">
        <h3 className="font-medium">Learning your task categories</h3>
        <p className="mt-2 text-sm text-muted-foreground">
          {settings.sampled} of {settings.sample_size} eligible requests collected. Keep sending traffic through the
          gateway. Categories will appear here for review.
        </p>
      </div>
    );
  if (settings?.phase === "error")
    return (
      <div className="space-y-3 rounded-lg border p-6">
        <p role="alert" className="text-destructive">
          {error || settings.error}
        </p>
        <Button disabled={busy} onClick={() => void save()}>
          Retry discovery
        </Button>
      </div>
    );

  const actionLabel = settings ? "Activate categories" : "Start discovery";
  return (
    <section className="max-w-3xl space-y-5 rounded-lg border bg-card p-6">
      <div>
        <h3 className="text-lg font-semibold">
          {settings ? "Review task categories" : "Understand where your LLM spend goes"}
        </h3>
        <p className="mt-1 text-sm text-muted-foreground">
          {settings
            ? "Edit the proposed categories before Jev starts classifying new traffic."
            : "Discover categories from a sample of new traffic, then classify subsequent requests in the background."}
        </p>
      </div>
      {settings?.phase === "review" ? (
        categories.map((category, index) => (
          <div key={category.id} className="space-y-2 rounded-md border p-3">
            <div className="flex gap-2">
              <Input
                aria-label={`Category ${index + 1} name`}
                value={category.name}
                onChange={(event) =>
                  setCategories(
                    categories.map((item) => (item.id === category.id ? { ...item, name: event.target.value } : item)),
                  )
                }
              />
              <Button
                variant="ghost"
                aria-label={`Remove ${category.name}`}
                onClick={() => setCategories(categories.filter((item) => item.id !== category.id))}
              >
                <X />
              </Button>
            </div>
            <Textarea
              aria-label={`Category ${index + 1} description`}
              value={category.description}
              onChange={(event) =>
                setCategories(
                  categories.map((item) =>
                    item.id === category.id ? { ...item, description: event.target.value } : item,
                  ),
                )
              }
            />
          </div>
        ))
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-1 text-sm">
              Sample size
              <Input
                type="number"
                min={5}
                max={500}
                value={sampleSize}
                onChange={(event) => setSampleSize(Number(event.target.value))}
              />
            </label>
            <label className="space-y-1 text-sm">
              Team ID (optional)
              <Input value={teamId} onChange={(event) => setTeamId(event.target.value)} placeholder="All teams" />
            </label>
            <label className="space-y-1 text-sm">
              Discovery model
              <Input
                value={discoveryModel}
                onChange={(event) => setDiscoveryModel(event.target.value)}
                placeholder="Gateway model name"
              />
            </label>
            <label className="space-y-1 text-sm">
              Jev model
              <Input
                value={classifierModel}
                onChange={(event) => setClassifierModel(event.target.value)}
                placeholder="Gateway Jev deployment name"
              />
            </label>
          </div>
          <label
            className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-7 text-sm"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              if (!busy) void addFiles(event.dataTransfer.files);
            }}
          >
            <Upload className="size-5 text-muted-foreground" />
            <span>Drop business context here, or choose files</span>
            <span className="text-xs text-muted-foreground">Optional · PDF, TXT, Markdown · 1 MiB per file</span>
            <input
              aria-label="Business context files"
              type="file"
              multiple
              accept=".pdf,.txt,.md"
              disabled={busy}
              className="sr-only"
              onChange={(event) => void addFiles(event.target.files)}
            />
          </label>
          {files.map((file, index) => (
            <div key={`${file.name}-${index}`} className="flex items-center justify-between text-sm">
              <span>{file.name} · Ready</span>
              <Button
                variant="ghost"
                aria-label={`Remove ${file.name}`}
                onClick={() => setFiles(files.filter((_, position) => position !== index))}
              >
                <X />
              </Button>
            </div>
          ))}
          <p className="text-xs text-muted-foreground">
            Eligible logged prompts and uploaded context are sent to your configured analysis models. Analysis incurs
            model costs. Requests without logged content remain unclassified.
          </p>
        </>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <Button
        disabled={
          busy || (settings?.phase === "review" ? categories.length === 0 : !discoveryModel || !classifierModel)
        }
        onClick={() => void save()}
      >
        {busy ? "Processing…" : actionLabel}
      </Button>
    </section>
  );
}

function fileContent(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") return reject(new Error("Unable to read file"));
      resolve(reader.result.slice(reader.result.indexOf(",") + 1));
    };
    reader.onerror = () => reject(new Error("Unable to read file"));
    reader.readAsDataURL(file);
  });
}

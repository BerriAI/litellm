"use client";

import { Shield } from "lucide-react";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const CONTENT_WIDTHS = ["w-24", "w-48", "w-60", "w-44", "w-52"];

export default function SSOSettingsLoadingSkeleton() {
  return (
    <Card role="status" aria-label="Loading SSO configuration">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-3">
          <Shield className="size-6 text-muted-foreground" />
          <div>
            <h3 className="text-lg font-semibold text-foreground">SSO Configuration</h3>
            <p className="text-sm text-muted-foreground">Manage Single Sign-On authentication settings</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-8 w-48" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="divide-y divide-border overflow-hidden rounded-md border border-border">
          {CONTENT_WIDTHS.map((width) => (
            <div key={width} className="grid grid-cols-3">
              <div className="bg-muted/50 px-4 py-3">
                <Skeleton className="h-4 w-20" />
              </div>
              <div className="col-span-2 px-4 py-3">
                <Skeleton className={`h-4 ${width}`} />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

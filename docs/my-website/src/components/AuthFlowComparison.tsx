import React, { useState, useEffect, useRef, useCallback } from "react";

const SPEED = 2;
const CHILD_STEP_MS = 350;

function DownArrow() {
  return (
    <svg width="20" height="24" viewBox="0 0 20 24" style={{ flexShrink: 0, margin: "4px 0" }}>
      <path d="M10 0 L10 18 M4 12 L10 18 L16 12" stroke="#555" strokeWidth="2" fill="none" />
    </svg>
  );
}

function StageBox({
  stage,
  active,
  expanded,
  childActiveCount,
}: {
  stage: any;
  active: boolean;
  expanded: boolean;
  childActiveCount: number;
}) {
  const isBottleneck = stage.bottleneck;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <div
        style={{
          padding: "8px 20px",
          borderRadius: 8,
          border: `2px solid ${active ? stage.color : "#333"}`,
          background: active ? `${stage.color}22` : "#1a1a1a",
          color: active ? stage.color : "#888",
          fontWeight: 600,
          fontSize: 13,
          fontFamily: "monospace",
          minWidth: 180,
          textAlign: "center",
          transition: "all 0.3s ease",
        }}
      >
        {stage.label}
        {isBottleneck && active && (
          <div style={{ fontSize: 10, color: "#ef4444", marginTop: 2, fontWeight: 400 }}>
            ~21s / 40K req
          </div>
        )}
      </div>
      {isBottleneck && expanded && stage.children && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 3,
            padding: "8px 12px",
            borderRadius: 6,
            border: "1px solid #ef444466",
            background: "#ef44440a",
            marginTop: 4,
            width: "100%",
            transition: "all 0.4s ease",
          }}
        >
          {stage.children.map((child: string, i: number) => {
            const isResolved = i < childActiveCount;
            const isCurrent = i === childActiveCount - 1;
            return (
              <div
                key={i}
                style={{
                  fontSize: 11,
                  fontFamily: "monospace",
                  color: isResolved
                    ? i === 0 ? "#f59e0b" : "#ef4444"
                    : "#555",
                  padding: "2px 8px",
                  borderRadius: 4,
                  background: isResolved
                    ? i === 0 ? "#f59e0b11" : "#ef44440a"
                    : "transparent",
                  transition: "all 0.25s ease",
                  transform: isCurrent ? "scale(1.05)" : "scale(1)",
                }}
              >
                {isResolved ? "\u2713 " : "\u00B7 "}
                {child}
              </div>
            );
          })}
        </div>
      )}
      {stage.annotation && active && (
        <div style={{ fontSize: 10, color: "#888", fontStyle: "italic", marginTop: 2 }}>
          {stage.annotation}
        </div>
      )}
    </div>
  );
}

function FlowWrapper({
  stages,
  label,
  baseDelayMs,
  baseBottleneckDelay,
}: {
  stages: any[];
  label: string;
  baseDelayMs: number;
  baseBottleneckDelay: number;
}) {
  const [activeIndex, setActiveIndex] = useState(-1);
  const [expanded, setExpanded] = useState(false);
  const [childActiveCount, setChildActiveCount] = useState(0);
  const [done, setDone] = useState(false);
  const [triggerKey, setTriggerKey] = useState(0);
  const timerRef = useRef<any>(null);

  const clearTimers = useCallback(() => {
    clearTimeout(timerRef.current);
  }, []);

  const run = useCallback(() => {
    setActiveIndex(-1);
    setExpanded(false);
    setChildActiveCount(0);
    setDone(false);
    let i = 0;

    function stepChildren(stage: any, childIdx: number, onComplete: () => void) {
      if (childIdx >= stage.children.length) {
        timerRef.current = setTimeout(onComplete, baseBottleneckDelay * SPEED);
        return;
      }
      setChildActiveCount(childIdx + 1);
      timerRef.current = setTimeout(
        () => stepChildren(stage, childIdx + 1, onComplete),
        CHILD_STEP_MS * SPEED
      );
    }

    function next() {
      if (i >= stages.length) {
        setDone(true);
        return;
      }
      setActiveIndex(i);
      const stage = stages[i];
      if (stage.bottleneck && stage.children) {
        timerRef.current = setTimeout(() => {
          setExpanded(true);
          stepChildren(stage, 0, () => {
            i++;
            next();
          });
        }, 300 * SPEED);
      } else {
        i++;
        timerRef.current = setTimeout(next, baseDelayMs * SPEED);
      }
    }

    timerRef.current = setTimeout(next, 400 * SPEED);
  }, [stages, baseDelayMs, baseBottleneckDelay]);

  useEffect(() => {
    run();
    return clearTimers;
  }, [triggerKey]);

  useEffect(() => {
    if (done) {
      const t = setTimeout(() => {
        setDone(false);
        setTriggerKey((k) => k + 1);
      }, 1500 * SPEED);
      return () => clearTimeout(t);
    }
  }, [done]);

  return (
    <div
      style={{
        background: "#111",
        borderRadius: 12,
        padding: 24,
        border: "1px solid #333",
        margin: "24px 0",
      }}
    >
      <div style={{ marginBottom: 12 }}>
        <span style={{ fontWeight: 700, fontSize: 14, color: "#ccc" }}>{label}</span>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 0,
        }}
      >
        {stages.map((stage, i) => (
          <React.Fragment key={i}>
            {i > 0 && <DownArrow />}
            <StageBox
              stage={stage}
              active={i <= activeIndex}
              expanded={expanded}
              childActiveCount={childActiveCount}
            />
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

const STAGES_BEFORE = [
  { label: "Client", color: "#6b7280", width: 80 },
  { label: "FastAPI Router", color: "#3b82f6", width: 120 },
  {
    label: "solve_dependencies()",
    color: "#ef4444",
    width: 200,
    bottleneck: true,
    children: [
      "user_api_key_auth()",
      "api_key (Security)",
      "azure_api_key (Security)",
      "anthropic_api_key (Security)",
      "google_ai_key (Security)",
      "azure_apim (Security)",
      "custom_key (Security)",
    ],
  },
  { label: "Route Handler", color: "#3b82f6", width: 120 },
  { label: "Response", color: "#22c55e", width: 80 },
];

const STAGES_AFTER = [
  { label: "Client", color: "#6b7280", width: 80 },
  {
    label: "Auth Middleware",
    color: "#22c55e",
    width: 140,
    annotation: "auth cached in request.state",
  },
  {
    label: "Route Handler",
    color: "#3b82f6",
    width: 120,
    annotation: "reads cached result",
  },
  { label: "Response", color: "#22c55e", width: 80 },
];

export function BeforeFlow() {
  return (
    <FlowWrapper
      stages={STAGES_BEFORE}
      label="Before: FastAPI Depends()"
      baseDelayMs={400}
      baseBottleneckDelay={400}
    />
  );
}

export function AfterFlow() {
  return (
    <FlowWrapper
      stages={STAGES_AFTER}
      label="After: ASGI Middleware"
      baseDelayMs={400}
      baseBottleneckDelay={0}
    />
  );
}

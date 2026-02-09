import React, { useState, useEffect, useCallback, useRef } from 'react';
import styles from './styles.module.css';

interface Stage {
  label: string;
  subtitle: string;
  code: string;
}

const STAGES: Stage[] = [
  {
    label: 'Request Wrapping',
    subtitle: '_CachedRequest',
    code: 'request = _CachedRequest(scope, receive)',
  },
  {
    label: 'Sync Event',
    subtitle: 'anyio.Event()',
    code: 'response_sent = anyio.Event()',
  },
  {
    label: 'Memory Stream',
    subtitle: 'create_memory_object_stream()',
    code: 'send_stream, recv_stream = anyio.create_memory_object_stream()',
  },
  {
    label: 'Task Group',
    subtitle: 'create_task_group()',
    code: 'async with anyio.create_task_group() as task_group:',
  },
  {
    label: 'Background Task',
    subtitle: 'task_group.start_soon(coro)',
    code: 'task_group.start_soon(coro)  # app runs in separate task',
  },
  {
    label: 'Nested Task Group',
    subtitle: 'receive_or_disconnect()',
    code: 'async with anyio.create_task_group() as task_group: ...',
  },
  {
    label: 'Response Wrapping',
    subtitle: '_StreamingResponse',
    code: 'response = _StreamingResponse(status_code=..., content=body_stream())',
  },
];

const INTERVAL_MS = 1200;
const PAUSE_MS = 600;

export default function BaseHTTPMiddlewareAnimation() {
  const [activeStage, setActiveStage] = useState(0);
  const [paused, setPaused] = useState(false);
  const [expandedStage, setExpandedStage] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (paused) return;

    const advance = () => {
      setActiveStage((prev) => {
        const next = (prev + 1) % STAGES.length;
        // If wrapping around, add extra pause
        if (next === 0) {
          timerRef.current = setTimeout(() => {
            timerRef.current = setTimeout(advance, INTERVAL_MS);
          }, PAUSE_MS);
          return next;
        }
        timerRef.current = setTimeout(advance, INTERVAL_MS);
        return next;
      });
    };

    timerRef.current = setTimeout(advance, INTERVAL_MS);
    return clearTimer;
  }, [paused, clearTimer]);

  const handleStageClick = (index: number) => {
    clearTimer();
    setPaused(true);
    setActiveStage(index);

    if (expandedStage === index) {
      // Close panel and resume
      setExpandedStage(null);
      setPaused(false);
    } else {
      setExpandedStage(index);
    }
  };

  return (
    <div className={styles.pipelineWrapper}>
      <div className={styles.pipelineLabel}>7 steps per request</div>
      <div className={styles.pipeline}>
        {STAGES.map((stage, i) => (
          <div className={styles.stageWrapper} key={i}>
            <div
              className={`${styles.stage} ${activeStage === i ? styles.stageActive : ''}`}
              onClick={() => handleStageClick(i)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') handleStageClick(i);
              }}
            >
              <div className={styles.stageNumber}>{i + 1}</div>
              <div className={styles.stageLabel}>{stage.label}</div>
              <div className={styles.stageSubtitle}>{stage.subtitle}</div>
            </div>
          </div>
        ))}
      </div>
      <div
        className={`${styles.codePanel} ${expandedStage !== null ? styles.codePanelOpen : ''}`}
      >
        {expandedStage !== null && (
          <pre className={styles.codePanelCode}>
            <code>{STAGES[expandedStage].code}</code>
          </pre>
        )}
      </div>
    </div>
  );
}

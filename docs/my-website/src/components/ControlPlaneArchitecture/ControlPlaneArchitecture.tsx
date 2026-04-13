import React from 'react';
import styles from './styles.module.css';

/* ────────────────────── Shared small pieces ────────────────────── */

function InfraChip({ color, label }: { color: string; label: string }) {
  const dotClass =
    color === 'green'
      ? styles.infraDotGreen
      : color === 'blue'
        ? styles.infraDotBlue
        : styles.infraDotOrange;

  return (
    <span className={styles.infraChip}>
      <span className={`${styles.infraDot} ${dotClass}`} />
      {label}
    </span>
  );
}

/* ────────────────────── Architecture tab ────────────────────── */

function ArchitectureView() {
  return (
    <div className={styles.diagram}>
      {/* User */}
      <div className={styles.userRow}>
        <div className={styles.userIcon}>&#128100;</div>
        <span className={styles.userLabel}>Admin</span>
      </div>

      <div className={styles.connectorDown} />

      {/* Control Plane */}
      <div className={`${styles.node} ${styles.nodeControlPlane}`}>
        <div className={styles.nodeHeader}>
          <span className={styles.nodeTitle}>Control Plane</span>
          <span className={`${styles.badge} ${styles.badgeBlue}`}>UI</span>
        </div>
        <div className={styles.nodeSubtitle}>cp.example.com</div>
        <div className={styles.infraRow}>
          <InfraChip color="green" label="Own DB" />
          <InfraChip color="orange" label="Own Redis" />
          <InfraChip color="blue" label="Own Key" />
        </div>
      </div>

      {/* Branch connector */}
      <div className={styles.connectorBranch}>
        <div className={`${styles.branchLeg} ${styles.branchLegLeft}`} />
        <div className={`${styles.branchLeg} ${styles.branchLegRight}`} />
      </div>

      {/* Workers */}
      <div className={styles.workersRow}>
        <div className={`${styles.node} ${styles.nodeWorker} ${styles.nodeWorkerA}`}>
          <div className={styles.nodeHeader}>
            <span className={styles.nodeTitle}>Worker A</span>
            <span className={`${styles.badge} ${styles.badgeGreen}`}>US East</span>
          </div>
          <div className={styles.nodeSubtitle}>worker-a.example.com</div>
          <div className={styles.infraRow}>
            <InfraChip color="green" label="Own DB" />
            <InfraChip color="orange" label="Own Redis" />
            <InfraChip color="blue" label="Own Key" />
          </div>
        </div>

        <div className={`${styles.node} ${styles.nodeWorker} ${styles.nodeWorkerB}`}>
          <div className={styles.nodeHeader}>
            <span className={styles.nodeTitle}>Worker B</span>
            <span className={`${styles.badge} ${styles.badgePurple}`}>EU West</span>
          </div>
          <div className={styles.nodeSubtitle}>worker-b.example.com</div>
          <div className={styles.infraRow}>
            <InfraChip color="green" label="Own DB" />
            <InfraChip color="orange" label="Own Redis" />
            <InfraChip color="blue" label="Own Key" />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ────────────────────── Main component ────────────────────── */

export default function ControlPlaneArchitecture() {
  return (
    <div className={styles.wrapper}>
      <ArchitectureView />
    </div>
  );
}

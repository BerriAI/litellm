import React from 'react';
import styles from './styles.module.css';

/* ────────────────────── Shared small pieces ────────────────────── */

function InfraBox({ icon, label, color }: { icon: string; label: string; color: 'green' | 'blue' | 'orange' }) {
  const colorClass =
    color === 'green'
      ? styles.infraBoxGreen
      : color === 'blue'
        ? styles.infraBoxBlue
        : styles.infraBoxOrange;

  return (
    <div className={`${styles.infraBox} ${colorClass}`}>
      <span className={styles.infraBoxIcon}>{icon}</span>
      <span className={styles.infraBoxLabel}>{label}</span>
    </div>
  );
}

/* ────────────────────── Worker column with infra ────────────────────── */

function WorkerColumn({
  name,
  region,
  subtitle,
  nodeClass,
  badgeClass,
}: {
  name: string;
  region: string;
  subtitle: string;
  nodeClass: string;
  badgeClass: string;
}) {
  return (
    <div className={styles.workerColumn}>
      <div className={`${styles.node} ${styles.nodeWorker} ${nodeClass}`}>
        <div className={styles.nodeHeader}>
          <span className={styles.nodeTitle}>{name}</span>
          <span className={`${styles.badge} ${badgeClass}`}>{region}</span>
        </div>
        <div className={styles.nodeSubtitle}>{subtitle}</div>
        <div className={styles.nodeCaption}>Handles LLM requests</div>
      </div>
      <div className={styles.infraStack}>
        <InfraBox icon="🗄" label="Own Database" color="green" />
        <InfraBox icon="⚡" label="Own Redis" color="orange" />
      </div>
    </div>
  );
}

/* ────────────────────── Architecture diagram ────────────────────── */

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
          <span className={`${styles.badge} ${styles.badgeBlue}`}>ADMIN UI ONLY</span>
        </div>
        <div className={styles.nodeSubtitle}>cp.example.com</div>
        <div className={styles.nodeCaption}>
          Not a router — does not proxy LLM requests.
          <br />
          Lets admins switch between workers to manage them.
        </div>
      </div>

      {/* Branch connector with label */}
      <div className={styles.connectorBranchLabeled}>
        <span className={styles.connectorLabel}>UI management only</span>
        <div className={styles.connectorBranch}>
          <div className={`${styles.branchLeg} ${styles.branchLegLeft}`} />
          <div className={`${styles.branchLeg} ${styles.branchLegRight}`} />
        </div>
      </div>

      {/* Workers */}
      <div className={styles.workersRow}>
        <WorkerColumn
          name="Worker A"
          region="US East"
          subtitle="worker-a.example.com"
          nodeClass={styles.nodeWorkerA}
          badgeClass={styles.badgeGreen}
        />
        <WorkerColumn
          name="Worker B"
          region="EU West"
          subtitle="worker-b.example.com"
          nodeClass={styles.nodeWorkerB}
          badgeClass={styles.badgePurple}
        />
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

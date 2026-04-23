import DashboardWebRTCTester from "../../../../ui/litellm-dashboard/src/components/WebRTCTester.jsx";

const LIGHT_MODE_OVERRIDES = `
.wrt-wrap {
  background: #1f2937;
  border: 1px solid #334155;
}
.wrt-toggle,
.wrt-toggle:hover {
  background: #111827;
}
.wrt-toggle-title,
.we-msg {
  color: #e2e8f0;
}
.wrt-toggle-sub,
.wrt-label,
.wrt-field label,
.wrt-flow-box,
.wrt-flow-arrow,
.wrt-meta-row span:first-child,
.wrt-header-title,
.wrt-tab,
.we-time {
  color: #94a3b8;
}
.wrt-body,
.wrt-sidebar,
.wrt-main,
.wrt-header,
.wrt-tabs,
.wrt-sdp-box,
.wrt-sdp-hdr,
.wrt-divider {
  border-color: #334155;
}
.wrt-header {
  background: #111827;
}
.wrt-field input,
.wrt-mic-btn,
.wrt-status-pill {
  background: #0b1220;
  border-color: #334155;
  color: #e2e8f0;
}
.wrt-field input:focus,
.wrt-btn-ghost:hover {
  border-color: #60a5fa;
}
.wrt-btn-ghost {
  background: #0b1220;
  border-color: #334155;
  color: #e2e8f0;
}
.wrt-log::-webkit-scrollbar-thumb {
  background: #475569;
}
.wrt-tab.active {
  color: #93c5fd;
  border-bottom-color: #93c5fd;
}
.wrt-empty,
.wrt-audio-status,
.wrt-meta-row span:last-child {
  color: #cbd5e1;
}
.wrt-sdp-dot {
  background: #475569;
}
.wrt-sdp-pane textarea {
  color: #e2e8f0;
}
`;

export default function WebRTCTester() {
  return (
    <>
      <DashboardWebRTCTester />
      <style>{LIGHT_MODE_OVERRIDES}</style>
    </>
  );
}

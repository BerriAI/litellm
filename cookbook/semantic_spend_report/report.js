"use strict";

(() => {
  const encoded = document.getElementById("report-data").textContent.trim();
  const data = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(encoded), c => c.charCodeAt(0))));
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const money = amount => "$" + Number(amount).toFixed(6);
  const spend = rows => rows.reduce((total, r) => total + Number(r.spend_usd), 0);
  const time = timestamp => new Date(timestamp).toISOString().slice(11, 19);
  const labels = new Map(data.classifications.map(c => [c.request_id, c]));
  const content = new Map(data.presentation.map(item => [item.scope_id, item]));
  const sessions = data.manifest.sessions;
  const names = new Map(sessions.map(s => [s.session_id, s.name]));
  const shortNames = { ci_cd_pipeline: "CI/CD", api_service_behavior: "API development", database_query_performance: "Database", documentation: "Docs", unknown: "Unknown", mixed: "Mixed" };
  const categoryName = id => shortNames[id] || data.categories.find(c => c.id === id)?.name || id;
  const palette = ["var(--chart-2)", "var(--chart-1)", "var(--chart-3)", "var(--chart-4)", "var(--muted-foreground)", "var(--chart-5)"];
  const colors = new Map(data.categories.map((c, i) => [c.id, palette[i % palette.length]]));
  const sessionColors = new Map(sessions.map((s, i) => [s.session_id, palette[i % palette.length]]));
  const rows = data.requests.map(r => ({ ...r, category: labels.get(r.request_id).category, at: Date.parse(r.started_at) }));
  const fullStart = Date.parse(data.manifest.start);
  const fullEnd = Date.parse(data.manifest.end);
  let state = { start: fullStart, end: fullEnd, user: "all", category: "all", view: "sessions", sort: "spend" };
  let lastFocus = null;

  function baseRows() {
    return rows.filter(r => r.at >= state.start && r.at <= state.end && (state.user === "all" || r.user_id === state.user));
  }

  function selectedRows() {
    return baseRows().filter(r => state.category === "all" || r.category === state.category);
  }

  function categoryTotals(list) {
    return data.categories.map(c => {
      const requests = list.filter(r => r.category === c.id);
      return { id: c.id, requests, spend: spend(requests), sessions: new Set(requests.map(r => r.session_id)).size };
    });
  }

  function setCategory(id) {
    state = { ...state, category: state.category === id ? "all" : id, view: "sessions" };
    render();
  }

  function renderTimeline(list) {
    clearChartHover();
    const duration = Math.max(1, state.end - state.start);
    const bucketMs = duration <= 60000 ? 5000 : duration <= 3600000 ? 60000 : 3600000;
    const bucketStart = Math.floor(state.start / bucketMs) * bucketMs;
    const count = Math.max(1, Math.ceil((state.end - bucketStart + 1) / bucketMs));
    const buckets = Array.from({ length: count }, (_, i) => {
      const start = bucketStart + i * bucketMs;
      const requests = list.filter(r => r.at >= start && r.at < start + bucketMs);
      return { start, requests, values: sessions.map(s => spend(requests.filter(r => r.session_id === s.session_id))) };
    });
    el("bucket-label").textContent = bucketMs === 5000 ? "5-second buckets · UTC" : bucketMs === 60000 ? "1-minute buckets · UTC" : "Hourly buckets · UTC";
    if (!list.length) {
      el("timeline").innerHTML = '<div class="empty">No requests in this selection</div>';
      el("session-legend").innerHTML = "";
      return;
    }
    const width = Math.max(280, el("timeline").clientWidth - 24), height = 224, left = 56, right = 18, top = 22, bottom = 34;
    const chartWidth = width - left - right, chartHeight = height - top - bottom;
    const highest = Math.max(...buckets.map(b => b.values.reduce((a, v) => a + v, 0))) * 1.18 || 0.01;
    const x = i => left + (i + 0.5) * chartWidth / count;
    const y = value => top + chartHeight * (1 - value / highest);
    const grid = Array.from({ length: 4 }, (_, i) => {
      const amount = highest * i / 3;
      return `<line x1="${left}" x2="${width - right}" y1="${y(amount)}" y2="${y(amount)}" stroke="var(--border)" stroke-dasharray="3 4"/><text x="${left - 10}" y="${y(amount) + 4}" text-anchor="end" fill="var(--muted-foreground)" font-size="10">$${amount.toFixed(3)}</text>`;
    }).join("");
    const areas = sessions.map((session, index) => {
      const lower = buckets.map(b => b.values.slice(0, index).reduce((a, v) => a + v, 0));
      const upper = buckets.map((b, i) => lower[i] + b.values[index]);
      const points = [...upper.map((v, i) => `${x(i)},${y(v)}`), ...lower.map((v, i) => `${x(i)},${y(v)}`).reverse()].join(" ");
      return `<polygon data-series="${esc(session.session_id)}" points="${points}" fill="${sessionColors.get(session.session_id)}" fill-opacity="0.56" stroke="${sessionColors.get(session.session_id)}" stroke-width="1.5"><title>${esc(session.name)} · ${money(spend(list.filter(r => r.session_id === session.session_id)))}</title></polygon>`;
    }).join("");
    const axis = buckets.map((b, i) => i % Math.max(1, Math.ceil(count / 5)) === 0 || i === count - 1
      ? `<text x="${x(i)}" y="${height - 9}" text-anchor="middle" fill="var(--muted-foreground)" font-size="10">${time(b.start)}</text>` : "").join("");
    const hits = buckets.map((b, i) => {
      const tooltip = `${time(b.start)}–${time(b.start + bucketMs)} UTC\n${money(spend(b.requests))} · ${b.requests.length} requests\n` + sessions.map((s, n) => `${s.name}: ${money(b.values[n])}`).join("\n");
      const detail = { heading: `${time(b.start)}–${time(b.start + bucketMs)} UTC`, total: spend(b.requests), count: b.requests.length,
        series: sessions.map((session, n) => ({ name: session.name, value: b.values[n], color: sessionColors.get(session.session_id) })).sort((a, z) => z.value - a.value) };
      return `<rect class="bucket-hit" data-tooltip-detail="${esc(JSON.stringify(detail))}" x="${left + i * chartWidth / count}" y="${top}" width="${chartWidth / count}" height="${chartHeight}" fill="transparent" tabindex="0" data-bucket-spend="${spend(b.requests)}" data-bucket-count="${b.requests.length}" data-tooltip="${esc(tooltip)}" aria-label="${esc(tooltip)}"></rect>`;
    }).join("");
    el("timeline").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Stacked session spend in time buckets" data-spend="${spend(list)}" data-request-count="${list.length}">${grid}${areas}${axis}${hits}</svg>`;
    el("session-legend").innerHTML = sessions.map(s => `<button data-open-session="${esc(s.session_id)}"><span class="dot" style="background:${sessionColors.get(s.session_id)}"></span>${esc(s.name)}</button>`).join("");
  }

  function renderShare(list) {
    const total = spend(list);
    const values = categoryTotals(list).sort((a, b) => b.spend - a.spend);
    el("share-bar").innerHTML = values.filter(v => v.spend).map(v => `<button data-category="${esc(v.id)}" style="width:${100 * v.spend / total}%;background:${colors.get(v.id)}" aria-label="Filter ${esc(categoryName(v.id))}" title="${esc(categoryName(v.id))}: ${money(v.spend)}"></button>`).join("");
    el("share-legend").innerHTML = values.map(v => `<button class="share-row ${state.category === v.id ? "selected" : ""}" data-category="${esc(v.id)}" aria-pressed="${state.category === v.id}"><span class="dot" style="background:${colors.get(v.id)}"></span><span class="share-name">${esc(categoryName(v.id))}</span><span class="share-amount">${money(v.spend)}</span><span class="share-percent">${total ? (100 * v.spend / total).toFixed(1) : "0.0"}%</span></button>`).join("");
  }

  function mixBar(list) {
    const total = spend(list);
    return `<div class="mini-bar" aria-label="Work area mix">${categoryTotals(list).filter(c => c.spend).map(c => `<span style="background:${colors.get(c.id)};width:${total ? 100 * c.spend / total : 0}%" title="${esc(categoryName(c.id))}: ${money(c.spend)}"></span>`).join("")}</div>`;
  }

  function renderRankings(list) {
    const isSessions = state.view === "sessions";
    const total = spend(list);
    const entities = isSessions ? sessions.map(s => ({ id: s.session_id, requests: list.filter(r => r.session_id === s.session_id) })) : categoryTotals(list);
    const ranked = entities.filter(e => e.requests.length).sort((a, b) => {
      const delta = state.sort === "spend" ? spend(b.requests) - spend(a.requests) : b.requests.length - a.requests.length;
      return delta || a.id.localeCompare(b.id);
    });
    el("ranking-head").innerHTML = `<tr><th class="rank">#</th><th>${isSessions ? "Session" : "Work area"}</th><th class="numeric-heading">Spend</th><th class="numeric-heading">Share</th><th class="numeric-heading">Requests</th><th>${isSessions ? "Work areas" : "Sessions"}</th><th></th></tr>`;
    el("ranking-body").innerHTML = ranked.map((entity, index) => {
      const value = spend(entity.requests);
      const item = content.get((isSessions ? "session:" : "category:") + entity.id);
      const title = isSessions ? item?.title || names.get(entity.id) : categoryName(entity.id);
      const color = isSessions ? sessionColors.get(entity.id) : colors.get(entity.id);
      const subtitle = isSessions ? `${names.get(entity.id)} · ${entity.requests[0].user_id}` : `${new Set(entity.requests.map(r => r.session_id)).size} contributing sessions`;
      return `<tr data-entity="${esc(entity.id)}" data-spend="${value}" data-count="${entity.requests.length}"><td class="rank">${index + 1}</td><td><div class="entity"><span class="entity-icon" aria-hidden="true"><span class="dot" style="background:${color};width:10px;height:10px"></span></span><div><button class="entity-name" data-open-${isSessions ? "session" : "category"}="${esc(entity.id)}">${esc(title)}</button><div class="entity-subtitle">${esc(subtitle)}</div></div></div></td><td class="number">${money(value)}</td><td class="number">${total ? (100 * value / total).toFixed(1) : "0.0"}%</td><td class="number">${entity.requests.length}</td><td>${isSessions ? mixBar(entity.requests) : new Set(entity.requests.map(r => r.session_id)).size}</td><td><button data-open-${isSessions ? "session" : "category"}="${esc(entity.id)}" aria-label="View ${esc(title)}">›</button></td></tr>`;
    }).join("") || '<tr><td colspan="7" class="empty">No requests match these filters</td></tr>';
    el("ranking-caption").textContent = `${ranked.length} ${isSessions ? "sessions" : "work areas"} · ${list.length} requests · ${money(total)} total`;
  }

  function render() {
    const base = baseRows();
    const selected = selectedRows();
    el("total-spend").textContent = money(spend(selected));
    el("total-spend").dataset.value = String(spend(selected));
    el("session-count").textContent = new Set(selected.map(r => r.session_id)).size;
    el("request-count").textContent = selected.length;
    el("spend-caption").textContent = state.category === "all" ? "All work areas · gateway costs" : categoryName(state.category) + " · gateway costs";
    el("overhead-spend").textContent = money(data.overhead.spend_usd);
    el("overhead-caption").textContent = `${data.overhead.request_count} report calls · separate from filters`;
    el("window-caption").textContent = `${new Date(state.start).toISOString().slice(0, 10)} · ${time(state.start)}–${time(state.end)} UTC`;
    el("category-tabs").hidden = state.view !== "sessions";
    el("category-tabs").innerHTML = [{ id: "all", name: "All work areas" }, ...data.categories].map(c => `<button data-category="${esc(c.id)}" aria-pressed="${state.category === c.id}">${esc(c.id === "all" ? c.name : categoryName(c.id))}</button>`).join("");
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === state.view;
      button.setAttribute("aria-pressed", String(active));
      button.setAttribute("aria-selected", String(active));
      button.toggleAttribute("data-active", active);
      button.tabIndex = 0;
    });
    renderTimeline(selected);
    renderShare(base);
    renderRankings(selected);
  }

  function evidenceDetails(request) {
    const classification = labels.get(request.request_id);
    return `<details class="request-details" id="evidence-${esc(request.request_id)}"><summary>${time(request.at)} UTC · ${esc(categoryName(request.category))} · ${money(request.spend_usd)}</summary><div class="request-meta">${esc(request.request_id)}<br>${esc(request.model)}<br>Jev proposed ${esc(categoryName(classification.proposed_category))} · confidence ${Number(classification.confidence).toFixed(2)}</div><h4>User instruction</h4><pre>${esc(request.instruction)}</pre><h4>Assistant response</h4><pre>${esc(request.response)}</pre></details>`;
  }

  function openDetails(kind, id) {
    const selected = selectedRows();
    const scoped = selected.filter(r => kind === "session" ? r.session_id === id : r.category === id);
    if (!scoped.length) return;
    const scopeKey = kind === "session" && state.category !== "all" ? `group:${id}:${state.category}` : `${kind}:${id}`;
    const item = content.get(scopeKey);
    const allowed = new Set(scoped.map(r => r.request_id));
    const complete = item && item.request_ids.every(rid => allowed.has(rid)) && item.request_ids.length === allowed.size;
    const bullets = (item?.bullets || []).filter(b => b.request_ids.every(rid => allowed.has(rid)));
    const quotes = (item?.quotes || []).filter(q => allowed.has(q.request_id));
    const citations = ids => ids.map((rid, i) => `<button class="citation" data-evidence="${esc(rid)}" aria-label="Open source request ${esc(rid)}">[${i + 1}]</button>`).join("");
    const title = kind === "category" ? categoryName(id) : content.get("session:" + id)?.title || names.get(id);
    el("drawer-title").textContent = title;
    el("drawer-eyebrow").textContent = kind === "session" ? "SESSION DETAIL" : "WORK AREA DETAIL";
    const related = kind === "category" ? sessions.map(s => ({ id: s.session_id, name: names.get(s.session_id), requests: scoped.filter(r => r.session_id === s.session_id) })) : categoryTotals(scoped).map(c => ({ ...c, name: categoryName(c.id) }));
    el("drawer-content").innerHTML = `<p class="overview">${esc(complete ? item.overview : `${scoped.length} requests in the selected window. Findings below retain only evidence within this selection.`)}</p><div class="drawer-metrics"><div><small>Selected spend</small><strong>${money(spend(scoped))}</strong></div><div><small>Requests</small><strong>${scoped.length}</strong></div><div><small>Selected share</small><strong>${(100 * spend(scoped) / spend(selected)).toFixed(1)}%</strong></div></div>${!complete ? '<p class="scope-warning">This is a partial window. The full-session overview is hidden, and only findings with all cited evidence in this window are shown.</p>' : ""}<h3>What happened</h3><ul class="findings">${bullets.map(b => `<li>${esc(b.text)} ${citations(b.request_ids)}</li>`).join("") || '<li>No complete finding is supported within this selection. Explore the request evidence below.</li>'}</ul><h3>Notable quotes</h3>${quotes.map(q => `<blockquote><span class="quote-label">${esc(q.speaker === "user" ? "User request" : "Assistant response")}</span><p>“${esc(q.text)}”</p><button class="quote-source" data-evidence="${esc(q.request_id)}">View source request ↗</button></blockquote>`).join("") || '<p class="overview">No selected quotes fall inside this window</p>'}<h3>${kind === "category" ? "Contributing sessions" : "Spend breakdown"}</h3>${related.filter(r => r.requests.length).map(r => `<div class="related"><button data-related-kind="${kind === "category" ? "session" : "category"}" data-related-id="${esc(r.id)}">${esc(r.name)} ↗</button><span>${money(spend(r.requests))}</span></div>`).join("")}<h3>Request evidence <span class="badge">${scoped.length}</span></h3>${scoped.map(evidenceDetails).join("")}<p class="request-meta">Synthetic proposals and drafts. No executed changes or measured ROI. Quotes are verified exact source excerpts; classification confidence is not measured accuracy.</p>`;
    if (!el("detail-drawer").open) {
      lastFocus = document.activeElement;
      el("detail-drawer").showModal();
    }
    el("detail-drawer").scrollTop = 0;
    el("close-drawer").focus({ preventScroll: true });
  }

  function syncDates() {
    el("start-filter").value = new Date(Math.floor(state.start / 1000) * 1000).toISOString().slice(0, 19);
    el("end-filter").value = new Date(Math.ceil(state.end / 1000) * 1000).toISOString().slice(0, 19);
  }

  function changeDates() {
    const start = Date.parse(el("start-filter").value + "Z");
    const end = Date.parse(el("end-filter").value + "Z");
    const valid = Number.isFinite(start) && Number.isFinite(end) && end >= start && end - start <= 31 * 86400000;
    el("filter-error").hidden = valid;
    el("filter-error").textContent = valid ? "" : "Choose a valid UTC interval of at most 31 days, with the end after the start";
    if (!valid) return;
    state = { ...state, start, end };
    el("period-filter").value = "custom";
    render();
  }

  document.addEventListener("click", event => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.dataset.category) setCategory(button.dataset.category);
    if (button.dataset.view) { state = { ...state, view: button.dataset.view, category: "all" }; render(); }
    if (button.dataset.openSession) openDetails("session", button.dataset.openSession);
    if (button.dataset.openCategory) openDetails("category", button.dataset.openCategory);
    if (button.dataset.relatedKind) {
      if (button.dataset.relatedKind === "category") { state = { ...state, category: button.dataset.relatedId }; render(); }
      openDetails(button.dataset.relatedKind, button.dataset.relatedId);
    }
    if (button.dataset.evidence) {
      const target = el("evidence-" + button.dataset.evidence);
      if (target) { target.open = true; target.scrollIntoView({ block: "center", behavior: "instant" }); target.querySelector("summary").focus(); }
    }
  });
  el("close-drawer").addEventListener("click", () => el("detail-drawer").close());
  el("detail-drawer").addEventListener("close", () => { lastFocus?.focus(); });
  el("detail-drawer").addEventListener("click", event => { if (event.target === el("detail-drawer") && event.clientX < el("detail-drawer").getBoundingClientRect().left) el("detail-drawer").close(); });
  el("sort-filter").addEventListener("change", event => { state = { ...state, sort: event.target.value }; render(); });
  el("user-filter").addEventListener("change", event => { state = { ...state, user: event.target.value }; render(); });
  el("period-filter").addEventListener("change", event => {
    const middle = fullStart + (fullEnd - fullStart) / 2;
    const range = event.target.value;
    if (range === "custom") { el("start-filter").focus(); return; }
    state = { ...state, start: range === "last" ? middle : fullStart, end: range === "first" ? middle : fullEnd };
    el("filter-error").hidden = true;
    syncDates(); render();
  });
  el("start-filter").addEventListener("change", changeDates);
  el("end-filter").addEventListener("change", changeDates);
  el("reset-filters").addEventListener("click", () => {
    state = { start: fullStart, end: fullEnd, user: "all", category: "all", view: "sessions", sort: "spend" };
    el("period-filter").value = "all"; el("user-filter").value = "all"; el("sort-filter").value = "spend";
    el("filter-error").hidden = true;
    syncDates(); render();
  });
  function clearChartHover() {
    el("chart-tooltip").hidden = true;
    document.querySelectorAll(".bucket-hit.active").forEach(node => node.classList.remove("active"));
    document.querySelectorAll("[data-series]").forEach(node => node.style.opacity = "");
    document.querySelectorAll("#session-legend button").forEach(node => node.classList.remove("highlighted"));
  }

  function showChartTooltip(target, clientX, clientY) {
    clearChartHover();
    const detail = JSON.parse(target.dataset.tooltipDetail);
    const tooltip = el("chart-tooltip");
    const total = detail.total;
    tooltip.innerHTML = `<div class="tooltip-date">${esc(detail.heading)}</div><div class="tooltip-panel">${detail.series.map(series => `<div class="tooltip-row"><span class="tooltip-marker" style="background:${series.color}"></span><span class="tooltip-name">${esc(series.name)}</span><strong>${money(series.value)}</strong></div>`).join("")}<div class="tooltip-total"><span>Total spend</span><strong>${money(total)}</strong></div><div class="tooltip-meta"><span>Model requests</span><strong>${detail.count}</strong></div></div>`;
    tooltip.hidden = false;
    target.classList.add("active");
    const bounds = tooltip.getBoundingClientRect();
    tooltip.style.left = Math.max(8, Math.min(clientX + 16, window.innerWidth - bounds.width - 8)) + "px";
    tooltip.style.top = Math.max(8, Math.min(clientY + 16, window.innerHeight - bounds.height - 8)) + "px";
  }

  document.addEventListener("pointermove", event => {
    const target = event.target.closest("[data-tooltip-detail]");
    if (target) {
      showChartTooltip(target, event.clientX, event.clientY);
      return;
    }
    clearChartHover();
    const legend = event.target.closest("#session-legend [data-open-session]");
    if (legend) {
      legend.classList.add("highlighted");
      document.querySelectorAll("[data-series]").forEach(node => node.style.opacity = node.dataset.series === legend.dataset.openSession ? "1" : "0.18");
    }
  });
  document.addEventListener("focusin", event => {
    const target = event.target.closest("[data-tooltip-detail]");
    if (!target) return;
    const bounds = target.getBoundingClientRect();
    showChartTooltip(target, bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  });
  document.addEventListener("focusout", clearChartHover);
  document.addEventListener("keydown", event => { if (event.key === "Escape") clearChartHover(); });
  document.addEventListener("scroll", clearChartHover, true);
  document.documentElement.addEventListener("pointerleave", clearChartHover);
  el("user-filter").innerHTML = '<option value="all">All selected users</option>' + [...new Set(rows.map(r => r.user_id))].map(user => `<option value="${esc(user)}">${esc(user)}</option>`).join("");
  window.addEventListener("resize", () => renderTimeline(selectedRows()));
  syncDates();
  render();
})();

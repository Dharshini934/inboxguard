(function () {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  // ---------- tabs ----------
  let resultsLoaded = false;
  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === t.dataset.tab));
    if (t.dataset.tab === "results" && !resultsLoaded) loadResults();
  }));

  // ---------- scan one message ----------
  $$(".chip").forEach((c) => c.addEventListener("click", () => { $("#msg").value = c.dataset.text; $("#msg").focus(); }));
  $("#clearBtn").addEventListener("click", () => { $("#msg").value = ""; $("#result").hidden = true; $("#scanErr").textContent = ""; });

  const FACTS = [
    ["n_chars", "Characters", (v) => v],
    ["n_words", "Words", (v) => v],
    ["upper_ratio", "Capital letters", (v) => (v * 100).toFixed(0) + "%"],
    ["digit_ratio", "Digits", (v) => (v * 100).toFixed(0) + "%"],
    ["exclamations", "Exclamation marks", (v) => v],
    ["has_url", "Contains a web link", (v) => (v ? "yes" : "no")],
    ["has_money", "Mentions an amount", (v) => (v ? "yes" : "no")],
    ["has_phone", "Contains a phone number", (v) => (v ? "yes" : "no")],
  ];

  async function scan() {
    const text = $("#msg").value.trim();
    $("#scanErr").textContent = "";
    if (!text) { $("#scanErr").textContent = "Type or paste a message first."; return; }
    const btn = $("#scanBtn"); btn.disabled = true; btn.textContent = "Scanning...";
    try {
      const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Something went wrong.");
      show(d);
    } catch (e) { $("#scanErr").textContent = e.message; }
    btn.disabled = false; btn.textContent = "Scan message";
  }
  $("#scanBtn").addEventListener("click", scan);

  function show(d) {
    const spam = d.label === "spam";
    $("#result").hidden = false;
    const st = $("#stamp"); st.textContent = spam ? "SPAM" : "CLEARED"; st.className = "stamp " + (spam ? "spam" : "ham");
    $("#probText").textContent = "Spam score " + (d.probability * 100).toFixed(1) + "%";
    const fill = $("#meterFill"); fill.className = "fill " + (spam ? "spam" : "");
    fill.style.width = (d.probability * 100) + "%";
    $("#meterThr").style.left = (d.threshold * 100) + "%";
    $("#thrText").textContent = "flagged above " + (d.threshold * 100).toFixed(0) + "%";

    const ul = $("#signals"); ul.innerHTML = "";
    const max = Math.max(0.01, ...d.signals.map((s) => Math.abs(s.weight)));
    d.signals.forEach((s) => {
      const li = document.createElement("li");
      const term = document.createElement("span"); term.className = "term"; term.textContent = s.term; term.title = s.term;
      const bar = document.createElement("div"); bar.className = "bar";
      const i = document.createElement("i"); i.className = s.weight > 0 ? "pos" : "neg";
      i.style.width = (Math.abs(s.weight) / max * 50) + "%"; bar.appendChild(i);
      const w = document.createElement("span"); w.className = "w"; w.textContent = (s.weight > 0 ? "+" : "") + s.weight.toFixed(2);
      li.append(term, bar, w); ul.appendChild(li);
    });
    $("#noSignals").hidden = d.signals.length > 0;

    const dl = $("#facts"); dl.innerHTML = "";
    FACTS.forEach(([k, label, fmt]) => {
      const dt = document.createElement("dt"); dt.textContent = label;
      const dd = document.createElement("dd"); dd.textContent = fmt(d.style[k]);
      dl.append(dt, dd);
    });
  }

  // ---------- model results ----------
  const pct = (v) => (v * 100).toFixed(1) + "%";
  async function loadResults() {
    const m = await (await fetch("/api/metrics")).json();
    resultsLoaded = true;
    const ds = m.dataset;
    $("#resultsLead").textContent = "Three algorithms were trained on " + ds.train.toLocaleString() + " messages and tested on " + ds.test.toLocaleString() + " unseen ones. " + m.best_model + " was chosen as the final model.";
    const facts = [[ds.messages.toLocaleString(), "unique messages"], [ds.spam.toLocaleString(), "spam"], [ds.ham.toLocaleString(), "normal (ham)"], [ds.duplicates_removed, "duplicates removed"]];
    $("#dataFacts").innerHTML = "";
    facts.forEach(([b, s]) => { const d = document.createElement("div"); d.className = "fact"; const bb = document.createElement("b"); bb.textContent = b; const ss = document.createElement("span"); ss.textContent = s; d.append(bb, ss); $("#dataFacts").appendChild(d); });

    const head = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Ham flagged as spam", "Spam missed"];
    const t = $("#modelTable"); t.innerHTML = "";
    const tr = document.createElement("tr"); head.forEach((h) => { const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); }); t.appendChild(tr);
    m.models.forEach((r) => {
      const row = document.createElement("tr"); if (r.model === m.best_model) row.className = "best";
      const cells = [r.model, pct(r.accuracy), pct(r.precision), pct(r.recall), r.f1.toFixed(3), r.roc_auc.toFixed(3), r.false_positives, r.false_negatives];
      cells.forEach((c, i) => { const td = document.createElement("td"); td.textContent = c; if (i > 0) td.className = "num"; if (i === 0 && r.model === m.best_model) { const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = "final model"; td.appendChild(tag); } row.appendChild(td); });
      t.appendChild(row);
    });

    const figs = [
      ["fig1_overview.png", "Class balance and message length: spam is the minority, and spam messages tend to be longer.", "wide"],
      ["fig2_style_signals.png", "Style signals. Spam carries more digits, links, amounts and phone numbers than normal messages.", "wide"],
      ["fig3_precision_recall.png", "Precision against recall. Dots mark the threshold chosen for each model.", ""],
      ["fig5_metrics.png", "Precision, recall and F1 on the test set.", ""],
      ["fig4_confusion.png", "Confusion matrices: how many messages were classified correctly and wrongly.", "wide"],
      ["fig6_top_terms.png", "Words that push a message towards spam (left) or towards normal (right).", "wide"],
    ];
    const box = $("#figs"); box.innerHTML = "";
    figs.forEach(([f, cap, cls]) => { const fg = document.createElement("figure"); fg.className = cls; const im = document.createElement("img"); im.src = "/reports/" + f; im.alt = cap; const fc = document.createElement("figcaption"); fc.textContent = cap; fg.append(im, fc); box.appendChild(fg); });
  }

  // ---------- batch ----------
  let batchRows = [];
  $("#batchBtn").addEventListener("click", async () => {
    $("#batchErr").textContent = "";
    const f = $("#csvFile").files[0];
    if (!f) { $("#batchErr").textContent = "Choose a CSV file first."; return; }
    const fd = new FormData(); fd.append("file", f);
    const btn = $("#batchBtn"); btn.disabled = true; btn.textContent = "Scanning...";
    try {
      const r = await fetch("/api/batch", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Something went wrong.");
      batchRows = d.rows;
      const s = $("#batchSummary"); s.innerHTML = "";
      [[d.total, "messages scanned"], [d.spam, "flagged as spam"], [d.total - d.spam, "cleared"]].forEach(([b, l]) => { const x = document.createElement("div"); x.className = "fact"; const bb = document.createElement("b"); bb.textContent = b; const ss = document.createElement("span"); ss.textContent = l; x.append(bb, ss); s.appendChild(x); });
      const t = $("#batchTable"); t.innerHTML = "";
      const hr = document.createElement("tr"); ["Verdict", "Spam score", "Message"].forEach((h) => { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); }); t.appendChild(hr);
      d.rows.slice(0, 200).forEach((r) => {
        const tr = document.createElement("tr");
        const a = document.createElement("td"); const tag = document.createElement("span"); tag.className = "tag " + (r.label === "spam" ? "spam" : ""); tag.textContent = r.label === "spam" ? "spam" : "cleared"; a.appendChild(tag);
        const b = document.createElement("td"); b.className = "num"; b.textContent = (r.probability * 100).toFixed(1) + "%";
        const c = document.createElement("td"); c.textContent = r.text.length > 160 ? r.text.slice(0, 160) + "..." : r.text;
        tr.append(a, b, c); t.appendChild(tr);
      });
      $("#dlBtn").hidden = false;
    } catch (e) { $("#batchErr").textContent = e.message; }
    btn.disabled = false; btn.textContent = "Scan file";
  });

  $("#dlBtn").addEventListener("click", () => {
    const esc = (v) => '"' + String(v).replace(/"/g, '""') + '"';
    const csv = "text,label,spam_probability\n" + batchRows.map((r) => [esc(r.text), r.label, r.probability].join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "inboxguard_results.csv"; a.click();
  });
})();

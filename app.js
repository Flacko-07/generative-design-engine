function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function fmt(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return parseCsv(await response.text());
}

function setStatus(text, state = 'ready') {
  const dot = document.querySelector('#status-dot');
  const statusEl = document.querySelector('#inference-status');
  if (statusEl) statusEl.textContent = text;
  if (dot) dot.className = 'status-dot ' + state;
}

async function render() {
  const [validation, training] = await Promise.all([
    loadCsv("validation_final.csv"),
    loadCsv("combined_final.csv"),
  ]);

  const laminar = training.filter((row) => row.flow_regime === "0").length;
  const turbulent = training.filter((row) => row.flow_regime === "1").length;
  const cds = training.map((row) => Number(row.Cd)).filter(Number.isFinite);

  document.querySelector("#training-rows").textContent = training.length.toLocaleString();
  document.querySelector("#laminar-count").textContent = laminar.toLocaleString();
  document.querySelector("#turbulent-count").textContent = turbulent.toLocaleString();
  document.querySelector("#drag-range").textContent = `${fmt(Math.min(...cds), 2)}\u2013${fmt(Math.max(...cds), 2)}`;

  document.querySelector("#validation-table").innerHTML = validation
    .map((row) => {
      const regime = row.regime === "0" ? "Laminar" : "Turbulent";
      const err = Number(row["relative_error_%"]);
      const errClass = err < 5 ? 'err-good' : err < 15 ? 'err-warn' : 'err-bad';
      return `<tr>
        <td>${fmt(row.target_Cd, 2)}</td>
        <td>${regime}</td>
        <td>${fmt(row.diameter, 4)}</td>
        <td>${fmt(row.actual_Cd, 4)}</td>
        <td class="${errClass}">${fmt(err, 1)}%</td>
      </tr>`;
    })
    .join("");
}

function renderDesigns(payload) {
  setStatus(`${payload.input.regime_label} \u2014 Cd ${fmt(payload.input.cd, 2)}`, 'ready');
  document.querySelector("#inference-latency").textContent = `API latency: ${payload.latency_ms} ms`;
  document.querySelector("#design-table").innerHTML = payload.designs
    .map((row) => `<tr>
      <td>${row.rank}</td>
      <td>${fmt(row.diameter, 5)}</td>
      <td>${fmt(row.x_center, 5)}</td>
      <td>${fmt(row.channel_height, 5)}</td>
      <td>${fmt(row.predicted_Cd, 4)}</td>
      <td>${fmt(row.Re, 0)}</td>
    </tr>`)
    .join("");
}

async function generateDesigns(event) {
  event.preventDefault();
  const params = new URLSearchParams(new FormData(event.currentTarget));
  setStatus("Generating\u2026", 'active');
  document.querySelector("#inference-latency").textContent = "API latency: running\u2026";

  const response = await fetch(`/api/generate?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Generation failed");
  renderDesigns(payload);
}

document.querySelector("#generate-form").addEventListener("submit", (event) => {
  generateDesigns(event).catch((error) => {
    setStatus(error.message, 'error');
    document.querySelector("#inference-latency").textContent = "API latency: \u2014";
  });
});

render().catch((error) => {
  document.querySelector("#training-rows").textContent = "Offline";
  document.querySelector("#validation-table").innerHTML =
    `<tr><td colspan="5" class="table-empty">${error.message}</td></tr>`;
});

document.querySelector("#generate-form").dispatchEvent(new Event("submit", { cancelable: true }));
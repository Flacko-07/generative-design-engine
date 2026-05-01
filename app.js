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
  document.querySelector("#drag-range").textContent = `${fmt(Math.min(...cds), 2)}-${fmt(Math.max(...cds), 2)}`;

  document.querySelector("#validation-table").innerHTML = validation
    .map((row) => {
      const regime = row.regime === "0" ? "Laminar" : "Turbulent";
      return `<tr>
        <td>${fmt(row.target_Cd, 2)}</td>
        <td>${regime}</td>
        <td>${fmt(row.diameter, 4)}</td>
        <td>${fmt(row.actual_Cd, 4)}</td>
        <td>${fmt(row["relative_error_%"], 1)}%</td>
      </tr>`;
    })
    .join("");
}

function renderDesigns(payload) {
  document.querySelector("#inference-status").textContent =
    `${payload.input.regime_label} candidates for Cd ${fmt(payload.input.cd, 2)}`;
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
  document.querySelector("#inference-status").textContent = "Generating";
  document.querySelector("#inference-latency").textContent = "API latency: running";

  const response = await fetch(`/api/generate?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Generation failed");
  renderDesigns(payload);
}

document.querySelector("#generate-form").addEventListener("submit", (event) => {
  generateDesigns(event).catch((error) => {
    document.querySelector("#inference-status").textContent = error.message;
    document.querySelector("#inference-latency").textContent = "API latency: -";
  });
});

render().catch((error) => {
  document.querySelector("#training-rows").textContent = "Offline";
  document.querySelector("#validation-table").innerHTML = `<tr><td colspan="5">${error.message}</td></tr>`;
});

document.querySelector("#generate-form").dispatchEvent(new Event("submit", { cancelable: true }));

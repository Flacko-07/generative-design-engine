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

  const errors = validation.map((row) => Number(row["relative_error_%"])).filter(Number.isFinite);
  const mean = errors.reduce((sum, value) => sum + value, 0) / errors.length;
  const laminar = training.filter((row) => row.flow_regime === "0").length;
  const turbulent = training.filter((row) => row.flow_regime === "1").length;
  const cds = training.map((row) => Number(row.Cd)).filter(Number.isFinite);

  document.querySelector("#mean-error").textContent = `${fmt(mean, 1)}%`;
  document.querySelector("#sample-count").textContent = `${validation.length} validation simulations`;
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

render().catch((error) => {
  document.querySelector("#mean-error").textContent = "Offline";
  document.querySelector("#sample-count").textContent = error.message;
});

const fs = require("fs");
const path = require("path");

const DATASET_PATH = path.join(process.cwd(), "combined_final.csv");
const rows = parseCsv(fs.readFileSync(DATASET_PATH, "utf8")).map((row) => ({
  diameter: Number(row.diameter),
  x_center: Number(row.x_center),
  Re: Number(row.Re),
  Cd: Number(row.Cd),
  channel_height: Number(row.channel_height),
  U_inlet: Number(row.U_inlet),
  blockage_ratio: Number(row.blockage_ratio),
  flow_regime: Number(row.flow_regime),
}));

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function asNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function candidateFromNeighbors(neighbors, targetCd, index) {
  const phase = index + 1;
  let weightTotal = 0;
  const mixed = {
    diameter: 0,
    x_center: 0,
    channel_height: 0,
    Re: 0,
    Cd: 0,
    U_inlet: 0,
    blockage_ratio: 0,
  };

  neighbors.forEach((row, neighborIndex) => {
    const distance = Math.abs(row.Cd - targetCd);
    const weight = 1 / (distance + 0.015 + neighborIndex * 0.002);
    weightTotal += weight;
    Object.keys(mixed).forEach((key) => {
      mixed[key] += row[key] * weight;
    });
  });

  Object.keys(mixed).forEach((key) => {
    mixed[key] /= weightTotal;
  });

  const closest = neighbors[0];
  const spread = Math.max(0.015, Math.abs(closest.Cd - targetCd) * 0.25);
  const diameterJitter = Math.sin(targetCd * 13.7 + phase) * spread;
  const xJitter = Math.cos(targetCd * 8.3 + phase * 0.7) * spread * 4;
  const hJitter = Math.sin(targetCd * 5.1 + phase * 1.3) * spread * 2;

  return {
    rank: index + 1,
    target_Cd: Number(targetCd.toFixed(4)),
    predicted_Cd: Number(mixed.Cd.toFixed(4)),
    diameter: Number(clamp(mixed.diameter + diameterJitter, 0.02, 0.5).toFixed(5)),
    x_center: Number(clamp(mixed.x_center + xJitter, 0.5, 4.0).toFixed(5)),
    channel_height: Number(clamp(mixed.channel_height + hJitter, 1.5, 4.0).toFixed(5)),
    Re: Number(Math.max(0, mixed.Re).toFixed(2)),
    U_inlet: Number(mixed.U_inlet.toFixed(4)),
    blockage_ratio: Number(clamp(mixed.blockage_ratio, 0, 1).toFixed(5)),
    nearest_training_Cd: Number(closest.Cd.toFixed(4)),
  };
}

module.exports = function handler(req, res) {
  try {
    const targetCd = asNumber(req.query.cd, 1.1);
    const regime = Math.round(asNumber(req.query.regime, 1));
    const count = Math.round(clamp(asNumber(req.query.count, 5), 1, 12));

    if (!Number.isFinite(targetCd) || targetCd <= 0) {
      res.status(400).json({ error: "cd must be a positive number" });
      return;
    }

    if (![0, 1].includes(regime)) {
      res.status(400).json({ error: "regime must be 0 for laminar or 1 for turbulent" });
      return;
    }

    const pool = rows
      .filter((row) => row.flow_regime === regime)
      .sort((a, b) => Math.abs(a.Cd - targetCd) - Math.abs(b.Cd - targetCd));

    const startedAt = performance.now();
    const designs = Array.from({ length: count }, (_, index) =>
      candidateFromNeighbors(pool.slice(index, index + 8), targetCd, index),
    );
    const elapsedMs = performance.now() - startedAt;

    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=86400");
    res.status(200).json({
      input: {
        cd: targetCd,
        regime,
        regime_label: regime === 0 ? "laminar" : "turbulent",
        count,
      },
      method: "regime-aware nearest-neighbor inference over combined_final.csv",
      latency_ms: Number(elapsedMs.toFixed(3)),
      designs,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

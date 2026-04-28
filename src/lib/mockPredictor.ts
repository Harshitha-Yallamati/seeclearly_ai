/**
 * Client-side mock predictor for offline UI testing.
 */

import type { PredictionResult } from "./api";

type DRStage = "No_DR" | "Mild" | "Moderate" | "Severe" | "PDR";

const STAGES: DRStage[] = ["No_DR", "Mild", "Moderate", "Severe", "PDR"];

const EXPLANATIONS: Record<DRStage, string> = {
  No_DR:
    "No signs of diabetic retinopathy were detected. The retinal image does not show clear microaneurysms, hemorrhages, or hard exudates.",
  Mild:
    "Mild non-proliferative diabetic retinopathy was detected. Small microaneurysms may be visible in the retina.",
  Moderate:
    "Moderate non-proliferative diabetic retinopathy was detected. The image suggests more widespread microaneurysms, hemorrhages, or exudates.",
  Severe:
    "Severe non-proliferative diabetic retinopathy was detected. The retinal findings may include extensive hemorrhages, venous changes, or IRMA.",
  PDR:
    "Proliferative diabetic retinopathy was detected. This stage can include new abnormal blood vessel growth and requires prompt specialist review.",
};

function generateMockHeatmapURI(): string {
  const cx = 40 + Math.random() * 35;
  const cy = 35 + Math.random() * 40;
  const radius = 18 + Math.random() * 18;

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="0 0 224 224">
      <rect width="224" height="224" fill="#030712" />
      <circle cx="112" cy="112" r="94" fill="#050816" stroke="#0f172a" stroke-width="2" />
      <circle cx="${(cx / 100) * 224}" cy="${(cy / 100) * 224}" r="${(radius / 100) * 224}" fill="url(#hotspot1)" />
      <circle cx="${((cx + 14) / 100) * 224}" cy="${((cy + 8) / 100) * 224}" r="${((radius * 0.45) / 100) * 224}" fill="url(#hotspot2)" />
      <defs>
        <radialGradient id="hotspot1">
          <stop offset="0%" stop-color="#ff2200" />
          <stop offset="35%" stop-color="#ff9f00" stop-opacity="0.9" />
          <stop offset="70%" stop-color="#ffe600" stop-opacity="0.55" />
          <stop offset="100%" stop-color="#0014ff" stop-opacity="0.05" />
        </radialGradient>
        <radialGradient id="hotspot2">
          <stop offset="0%" stop-color="#fff6a5" stop-opacity="0.95" />
          <stop offset="55%" stop-color="#ff5a00" stop-opacity="0.45" />
          <stop offset="100%" stop-color="#0014ff" stop-opacity="0" />
        </radialGradient>
      </defs>
    </svg>
  `.trim();

  return `data:image/svg+xml;base64,${btoa(svg)}`;
}

function buildMockProbabilities(stageIndex: number): Record<string, number> {
  const raw = STAGES.map((_, index) =>
    index === stageIndex ? 4 + Math.random() * 2.5 : 0.15 + Math.random() * 0.8,
  );
  const total = raw.reduce((sum, value) => sum + value, 0);

  return Object.fromEntries(
    STAGES.map((stage, index) => [
      stage,
      Math.round((raw[index] / total) * 10000) / 10000,
    ]),
  );
}

export async function getMockPrediction(file: File): Promise<PredictionResult> {
  await new Promise((resolve) => setTimeout(resolve, 2200 + Math.random() * 1000));

  const stageIndex = (file.name.length + file.size) % 5;
  const label = STAGES[stageIndex];
  const probabilities = buildMockProbabilities(stageIndex);
  const confidence = Math.max(...Object.values(probabilities));

  let explanation = EXPLANATIONS[label];
  if (confidence < 0.75) {
    explanation += `\n\nLow confidence (${(confidence * 100).toFixed(0)}%). Please confirm this screening result with an ophthalmologist.`;
  }

  const heatmap = generateMockHeatmapURI();

  return {
    label,
    severity_index: stageIndex,
    confidence,
    probabilities,
    all_probabilities: probabilities,
    explanation,
    heatmap,
    overlay: heatmap,
    preprocessing: [
      "Decoded image and resized to 224x224 pixels",
      "Scaled pixels to the training-time range (0-1)",
      "Generated fallback probabilities for offline UI testing",
      "Generated fallback explainability heatmap",
    ],
    is_mock: true,
    heatmap_is_mock: true,
    needs_doctor: stageIndex >= 2 || confidence < 0.75,
    gradcam_layer: null,
    gradcam_method: "mock",
  };
}

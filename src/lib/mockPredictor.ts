/**
 * SeeClearly AI — Mock Predictor
 *
 * Client-side fallback when the Flask backend is unavailable.
 * Generates realistic-looking predictions for UI development/testing.
 */

import type { PredictionResult } from "./api";

type DRStage = "No_DR" | "Mild" | "Moderate" | "Severe" | "PDR";

const STAGES: DRStage[] = ["No_DR", "Mild", "Moderate", "Severe", "PDR"];

const EXPLANATIONS: Record<DRStage, string> = {
  No_DR:
    "No signs of Diabetic Retinopathy detected. The retinal blood vessels appear normal with no visible microaneurysms, hemorrhages, or exudates.",
  Mild:
    "Mild Non-Proliferative Diabetic Retinopathy (NPDR) detected. Small microaneurysms (tiny red dots) may be present. These are swollen areas in the small blood vessels of the retina.",
  Moderate:
    "Moderate NPDR detected. Multiple microaneurysms, some dot/blot hemorrhages, and possible hard exudates (yellow lipid deposits) are visible. Blood vessels may show signs of blockage.",
  Severe:
    "Severe NPDR detected. Significant retinal hemorrhages in all four quadrants, venous beading, and intraretinal microvascular abnormalities (IRMA). High risk of progression to proliferative stage.",
  PDR:
    "Proliferative Diabetic Retinopathy (PDR) detected. Abnormal new blood vessel growth (neovascularization) is present. These fragile vessels can leak blood into the vitreous, causing severe vision loss. URGENT medical intervention recommended.",
};

/**
 * Generate a realistic mock heatmap as a colored SVG data URI.
 */
function generateMockHeatmapURI(): string {
  const cx = 40 + Math.random() * 40;
  const cy = 40 + Math.random() * 40;
  const r = 15 + Math.random() * 20;

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="0 0 224 224">
      <rect width="224" height="224" fill="#1a1a2e"/>
      <circle cx="${(cx / 100) * 224}" cy="${(cy / 100) * 224}" r="${(r / 100) * 224}"
        fill="url(#hg)" opacity="0.7"/>
      <circle cx="${((cx + 15) / 100) * 224}" cy="${((cy - 10) / 100) * 224}" r="${((r * 0.5) / 100) * 224}"
        fill="url(#hg2)" opacity="0.5"/>
      <defs>
        <radialGradient id="hg">
          <stop offset="0%" stop-color="#ff4444"/>
          <stop offset="40%" stop-color="#ff8800" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="#0066ff" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="hg2">
          <stop offset="0%" stop-color="#ffcc00"/>
          <stop offset="60%" stop-color="#ff6600" stop-opacity="0.4"/>
          <stop offset="100%" stop-color="#0044aa" stop-opacity="0"/>
        </radialGradient>
      </defs>
    </svg>
  `.trim();

  return `data:image/svg+xml;base64,${btoa(svg)}`;
}

/**
 * Generate a mock prediction for UI testing.
 */
export async function getMockPrediction(file: File): Promise<PredictionResult> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 2500 + Math.random() * 1500));

  // Deterministic stage selection based on file properties
  const stageIndex = (file.name.length + file.size) % 5;
  const label = STAGES[stageIndex];
  const confidence = 0.72 + Math.random() * 0.25;

  // Generate fake probability distribution
  const probs: Record<string, number> = {};
  let remaining = 1 - confidence;
  STAGES.forEach((s, i) => {
    if (i === stageIndex) {
      probs[s] = Math.round(confidence * 10000) / 10000;
    } else {
      const share = remaining * (0.1 + Math.random() * 0.3);
      probs[s] = Math.round(share * 10000) / 10000;
      remaining -= share;
    }
  });

  let explanation = EXPLANATIONS[label];
  if (confidence < 0.75) {
    explanation += `\n\n⚠️ Low confidence (${(confidence * 100).toFixed(0)}%). The model has significant uncertainty. Please consult an ophthalmologist.`;
  }

  return {
    label,
    severity_index: stageIndex,
    confidence: Math.round(confidence * 10000) / 10000,
    all_probabilities: probs,
    explanation,
    heatmap: generateMockHeatmapURI(),
    preprocessing: [
      "Resized to 224×224",
      "Normalized pixel values (0–1)",
      "Applied EfficientNetB3 feature extraction",
      "Generated Grad-CAM heatmap overlay",
    ],
    is_mock: true,
    needs_doctor: stageIndex >= 2 || confidence < 0.75,
  };
}

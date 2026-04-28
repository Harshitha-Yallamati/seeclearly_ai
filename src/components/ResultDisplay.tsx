import { AlertTriangle, Info } from "lucide-react";

import type { PredictionResult } from "@/lib/api";

import HeatmapViewer from "./HeatmapViewer";
import SeverityGauge from "./SeverityGauge";

interface ResultDisplayProps {
  result: PredictionResult;
  originalPreview: string;
}

const STAGE_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "PDR"];

const formatStageName = (label: string) => label.replace("_", " ");

const ResultDisplay = ({ result, originalPreview }: ResultDisplayProps) => {
  const probabilities = result.probabilities ?? result.all_probabilities ?? {};
  const orderedProbabilities = STAGE_ORDER.filter(
    (stage) => typeof probabilities[stage] === "number",
  ).map((stage) => ({
    stage,
    value: probabilities[stage],
  }));

  return (
    <div className="space-y-5">
      {result.is_mock && (
        <div className="animate-fade-in-up flex items-center gap-2 rounded-lg border border-orange-500/20 bg-orange-500/10 px-3 py-2">
          <Info className="h-4 w-4 shrink-0 text-orange-400" />
          <p className="text-xs font-medium text-orange-300">
            Running in demo mode. Predictions are simulated because a trained
            backend model was not available.
          </p>
        </div>
      )}

      <SeverityGauge
        severityIndex={result.severity_index}
        confidence={result.confidence}
        label={formatStageName(result.label)}
      />

      {orderedProbabilities.length > 0 && (
        <div
          className="animate-fade-in-up rounded-xl p-4 glass"
          style={{ opacity: 0, animationFillMode: "forwards" }}
        >
          <h3 className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Class Probabilities
          </h3>

          <div className="space-y-2.5">
            {orderedProbabilities.map(({ stage, value }) => (
              <div key={stage} className="space-y-1">
                <div className="flex items-center justify-between gap-4 text-xs">
                  <span className="font-semibold text-foreground/90">
                    {formatStageName(stage)}
                  </span>
                  <span className="font-mono text-muted-foreground">
                    {(value * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-muted/50">
                  <div
                    className={`h-full rounded-full ${
                      stage === result.label ? "bg-primary" : "bg-foreground/20"
                    }`}
                    style={{ width: `${Math.max(0, Math.min(value, 1)) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
            Confidence is the maximum softmax probability across the five DR
            classes, so it always matches the highest bar above.
          </p>
        </div>
      )}

      {result.heatmap && (
        <div
          className="animate-fade-in-up delay-200"
          style={{ opacity: 0, animationFillMode: "forwards" }}
        >
          <h3 className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Explainability
          </h3>
          <HeatmapViewer
            originalPreview={originalPreview}
            heatmapBase64={result.heatmap}
            overlayBase64={result.overlay}
          />
          {(result.gradcam_method || result.gradcam_layer) && (
            <p className="mt-2 text-[11px] text-muted-foreground">
              {[
                result.gradcam_method
                  ? `Method: ${String(result.gradcam_method).toUpperCase()}`
                  : null,
                result.gradcam_layer
                  ? `Layer: ${result.gradcam_layer}`
                  : null,
              ]
                .filter(Boolean)
                .join(" | ")}
            </p>
          )}
        </div>
      )}

      {result.explanation && (
        <div
          className="animate-fade-in-up rounded-xl p-4 glass delay-300"
          style={{ opacity: 0, animationFillMode: "forwards" }}
        >
          <h3 className="mb-2 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Clinical Explanation
          </h3>
          <p className="whitespace-pre-line text-sm leading-relaxed text-foreground/90">
            {result.explanation}
          </p>
        </div>
      )}

      <div
        className="animate-fade-in-up rounded-xl p-4 glass delay-400"
        style={{ opacity: 0, animationFillMode: "forwards" }}
      >
        <h3 className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
          Processing Pipeline
        </h3>
        <ol className="space-y-1.5">
          {result.preprocessing.map((step, index) => (
            <li
              key={index}
              className="flex items-start gap-2.5 text-sm text-foreground/80"
            >
              <span className="mt-0.5 w-4 shrink-0 font-mono text-[10px] font-bold text-primary">
                {index + 1}.
              </span>
              {step}
            </li>
          ))}
        </ol>
      </div>

      {result.needs_doctor && (
        <div
          className="animate-fade-in-up delay-500 flex items-start gap-3 rounded-xl border border-destructive/20 bg-destructive/10 p-4"
          style={{ opacity: 0, animationFillMode: "forwards" }}
        >
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
          <div>
            <p className="text-sm font-bold text-destructive">
              Medical consultation recommended
            </p>
            <p className="mt-1 text-xs leading-relaxed text-destructive/80">
              {result.severity_index >= 2
                ? "Moderate-to-severe retinopathy was detected. Please arrange an ophthalmology review as soon as possible."
                : "The model confidence is below the clinical comfort threshold. An ophthalmologist should verify the result with a full retinal exam."}
            </p>
          </div>
        </div>
      )}

      <p className="pt-2 text-center text-[10px] italic text-muted-foreground/50">
        This is an AI-assisted screening tool for educational and research use
        only. It is not a medical diagnosis.
      </p>
    </div>
  );
};

export default ResultDisplay;

import { AlertTriangle, Info } from "lucide-react";
import type { PredictionResult } from "@/lib/api";
import SeverityGauge from "./SeverityGauge";
import HeatmapViewer from "./HeatmapViewer";

interface ResultDisplayProps {
  result: PredictionResult;
  originalPreview: string;
}

const ResultDisplay = ({ result, originalPreview }: ResultDisplayProps) => {
  return (
    <div className="space-y-5">
      {/* Mock mode badge */}
      {result.is_mock && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-orange-500/10 border border-orange-500/20 animate-fade-in-up">
          <Info className="w-4 h-4 text-orange-400 shrink-0" />
          <p className="text-xs text-orange-300 font-medium">
            Running in demo mode — predictions are simulated. Start the Flask backend for real model inference.
          </p>
        </div>
      )}

      {/* Severity Gauge */}
      <SeverityGauge
        severityIndex={result.severity_index}
        confidence={result.confidence}
        label={result.label}
      />

      {/* Grad-CAM Heatmap */}
      {result.heatmap && (
        <div className="animate-fade-in-up delay-200" style={{ opacity: 0, animationFillMode: "forwards" }}>
          <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
            Explainability — Grad-CAM
          </h3>
          <HeatmapViewer
            originalPreview={originalPreview}
            heatmapBase64={result.heatmap}
          />
        </div>
      )}

      {/* Medical Explanation */}
      {result.explanation && (
        <div className="animate-fade-in-up delay-300 glass rounded-xl p-4" style={{ opacity: 0, animationFillMode: "forwards" }}>
          <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-primary" />
            Clinical Explanation
          </h3>
          <p className="text-sm text-foreground/90 leading-relaxed whitespace-pre-line">
            {result.explanation}
          </p>
        </div>
      )}

      {/* Preprocessing Steps */}
      <div className="animate-fade-in-up delay-400 glass rounded-xl p-4" style={{ opacity: 0, animationFillMode: "forwards" }}>
        <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          Preprocessing Pipeline
        </h3>
        <ol className="space-y-1.5">
          {result.preprocessing.map((step, i) => (
            <li key={i} className="flex items-start gap-2.5 text-sm text-foreground/80">
              <span className="text-[10px] font-mono text-primary font-bold mt-0.5 w-4 shrink-0">
                {i + 1}.
              </span>
              {step}
            </li>
          ))}
        </ol>
      </div>

      {/* Doctor Warning */}
      {result.needs_doctor && (
        <div className="animate-fade-in-up delay-500 flex items-start gap-3 p-4 rounded-xl bg-destructive/10 border border-destructive/20" style={{ opacity: 0, animationFillMode: "forwards" }}>
          <AlertTriangle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-bold text-destructive">
              Medical Consultation Recommended
            </p>
            <p className="text-xs text-destructive/80 mt-1 leading-relaxed">
              {result.severity_index >= 2
                ? "Moderate-to-severe retinopathy detected. Please schedule an appointment with an ophthalmologist as soon as possible."
                : "Model confidence is below threshold. An ophthalmologist should verify this result with a comprehensive dilated eye exam."}
            </p>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-[10px] text-muted-foreground/50 text-center italic pt-2">
        This is an AI-assisted screening tool for educational and research purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
      </p>
    </div>
  );
};

export default ResultDisplay;

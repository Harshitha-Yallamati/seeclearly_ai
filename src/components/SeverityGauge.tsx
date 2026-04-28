import { useEffect, useState } from "react";

interface SeverityGaugeProps {
  severityIndex: number;
  confidence: number;
  label: string;
}

const SEVERITY_CONFIG = [
  {
    name: "No DR",
    color: "#22c55e",
    bgColor: "rgba(34,197,94,0.1)",
    borderColor: "rgba(34,197,94,0.3)",
  },
  {
    name: "Mild",
    color: "#eab308",
    bgColor: "rgba(234,179,8,0.1)",
    borderColor: "rgba(234,179,8,0.3)",
  },
  {
    name: "Moderate",
    color: "#f97316",
    bgColor: "rgba(249,115,22,0.1)",
    borderColor: "rgba(249,115,22,0.3)",
  },
  {
    name: "Severe",
    color: "#ef4444",
    bgColor: "rgba(239,68,68,0.1)",
    borderColor: "rgba(239,68,68,0.3)",
  },
  {
    name: "PDR",
    color: "#dc2626",
    bgColor: "rgba(220,38,38,0.1)",
    borderColor: "rgba(220,38,38,0.3)",
  },
];

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

const SeverityGauge = ({
  severityIndex,
  confidence,
  label,
}: SeverityGaugeProps) => {
  const [animatedProgress, setAnimatedProgress] = useState(0);
  const [animatedConfidence, setAnimatedConfidence] = useState(0);

  const clampedSeverity = clamp(Math.round(severityIndex), 0, 4);
  const clampedConfidence = clamp(confidence, 0, 1);
  const config = SEVERITY_CONFIG[clampedSeverity] || SEVERITY_CONFIG[0];
  const progressPercent = (clampedSeverity / 4) * 100;

  useEffect(() => {
    const timer1 = setTimeout(() => setAnimatedProgress(progressPercent), 100);
    const timer2 = setTimeout(
      () => setAnimatedConfidence(clampedConfidence * 100),
      180,
    );

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [clampedConfidence, progressPercent]);

  return (
    <div
      className="animate-fade-in-up rounded-xl border-2 p-5 transition-all duration-500"
      style={{
        backgroundColor: config.bgColor,
        borderColor: config.borderColor,
      }}
    >
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <p className="text-lg font-bold" style={{ color: config.color }}>
            {label}
          </p>
          <p className="mt-0.5 text-xs font-medium text-muted-foreground">
            Severity level {clampedSeverity} of 4
          </p>
        </div>

        <div className="text-right">
          <p
            className="text-2xl font-black tabular-nums"
            style={{ color: config.color }}
          >
            {animatedConfidence.toFixed(1)}%
          </p>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Predicted class confidence
          </p>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          <span>No DR</span>
          <span>PDR</span>
        </div>

        <div className="relative h-2.5 overflow-hidden rounded-full bg-muted/50">
          <div
            className="h-full rounded-full transition-all duration-1000 ease-out"
            style={{
              width: `${animatedProgress}%`,
              background:
                "linear-gradient(90deg, #22c55e, #eab308, #f97316, #ef4444, #dc2626)",
            }}
          />
        </div>

        <div className="flex justify-between px-1">
          {SEVERITY_CONFIG.map((stage, index) => (
            <div
              key={stage.name}
              className={`h-2.5 w-2.5 rounded-full border-2 transition-all duration-300 ${
                index === clampedSeverity ? "scale-125 shadow-lg" : "opacity-40"
              }`}
              style={{
                backgroundColor:
                  index === clampedSeverity ? stage.color : "transparent",
                borderColor: stage.color,
              }}
              title={stage.name}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default SeverityGauge;

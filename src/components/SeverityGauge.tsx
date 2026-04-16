import { useEffect, useState } from "react";

interface SeverityGaugeProps {
  severityIndex: number; // 0-4
  confidence: number;    // 0-1
  label: string;
}

const SEVERITY_CONFIG = [
  { name: "No DR",    color: "#22c55e", bgColor: "rgba(34,197,94,0.1)",    borderColor: "rgba(34,197,94,0.3)" },
  { name: "Mild",     color: "#eab308", bgColor: "rgba(234,179,8,0.1)",    borderColor: "rgba(234,179,8,0.3)" },
  { name: "Moderate", color: "#f97316", bgColor: "rgba(249,115,22,0.1)",   borderColor: "rgba(249,115,22,0.3)" },
  { name: "Severe",   color: "#ef4444", bgColor: "rgba(239,68,68,0.1)",    borderColor: "rgba(239,68,68,0.3)" },
  { name: "PDR",      color: "#dc2626", bgColor: "rgba(220,38,38,0.1)",    borderColor: "rgba(220,38,38,0.3)" },
];

const SeverityGauge = ({ severityIndex, confidence, label }: SeverityGaugeProps) => {
  const [animatedProgress, setAnimatedProgress] = useState(0);
  const [animatedConfidence, setAnimatedConfidence] = useState(0);

  const config = SEVERITY_CONFIG[severityIndex] || SEVERITY_CONFIG[0];
  const progressPercent = ((severityIndex + 1) / 5) * 100;

  useEffect(() => {
    const timer1 = setTimeout(() => setAnimatedProgress(progressPercent), 100);
    const timer2 = setTimeout(() => setAnimatedConfidence(confidence * 100), 200);
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [progressPercent, confidence]);

  return (
    <div
      className="p-5 rounded-xl border-2 transition-all duration-500 animate-fade-in-up"
      style={{
        backgroundColor: config.bgColor,
        borderColor: config.borderColor,
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-lg font-bold" style={{ color: config.color }}>
            {label}
          </p>
          <p className="text-xs text-muted-foreground font-medium mt-0.5">
            Severity Level {severityIndex} of 4
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-black tabular-nums" style={{ color: config.color }}>
            {animatedConfidence.toFixed(1)}%
          </p>
          <p className="text-[10px] text-muted-foreground font-semibold uppercase tracking-wider">
            Confidence
          </p>
        </div>
      </div>

      {/* Severity Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
          <span>No DR</span>
          <span>PDR</span>
        </div>
        <div className="h-2.5 rounded-full bg-muted/50 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-1000 ease-out"
            style={{
              width: `${animatedProgress}%`,
              background: `linear-gradient(90deg, #22c55e, #eab308, #f97316, #ef4444, #dc2626)`,
            }}
          />
        </div>

        {/* Stage indicators */}
        <div className="flex justify-between px-1">
          {SEVERITY_CONFIG.map((s, i) => (
            <div
              key={i}
              className={`w-2.5 h-2.5 rounded-full border-2 transition-all duration-300 ${
                i === severityIndex ? "scale-125 shadow-lg" : "opacity-40"
              }`}
              style={{
                backgroundColor: i === severityIndex ? s.color : "transparent",
                borderColor: s.color,
              }}
              title={s.name}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default SeverityGauge;

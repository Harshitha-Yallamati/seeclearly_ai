import { useEffect, useState } from "react";

interface AnalysisAnimationProps {
  imagePreview: string;
}

const STEPS = [
  { label: "Preprocessing image", detail: "Resize → Normalize → RGB conversion", duration: 800 },
  { label: "Extracting features", detail: "EfficientNetB3 feature extraction", duration: 1200 },
  { label: "Running classification", detail: "5-class softmax prediction", duration: 800 },
  { label: "Generating heatmap", detail: "Grad-CAM attention mapping", duration: 1000 },
];

const AnalysisAnimation = ({ imagePreview }: AnalysisAnimationProps) => {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    let step = 0;
    const advanceStep = () => {
      step++;
      if (step < STEPS.length) {
        setCurrentStep(step);
        setTimeout(advanceStep, STEPS[step].duration);
      }
    };
    setTimeout(advanceStep, STEPS[0].duration);
    // eslint-disable-next-line
  }, []);

  return (
    <div className="space-y-4 animate-fade-in-up">
      {/* Image with scan effect */}
      <div className="relative rounded-xl overflow-hidden border border-primary/30 glass-glow">
        <img
          src={imagePreview}
          alt="Analyzing retinal fundus"
          className="w-full h-56 object-contain bg-black/20 opacity-80"
        />

        {/* Scanning line */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent opacity-80 animate-scan-line" />
        </div>

        {/* Corner brackets */}
        <div className="absolute inset-0 pointer-events-none p-3">
          <div className="absolute top-3 left-3 w-6 h-6 border-t-2 border-l-2 border-primary/60 rounded-tl-lg" />
          <div className="absolute top-3 right-3 w-6 h-6 border-t-2 border-r-2 border-primary/60 rounded-tr-lg" />
          <div className="absolute bottom-3 left-3 w-6 h-6 border-b-2 border-l-2 border-primary/60 rounded-bl-lg" />
          <div className="absolute bottom-3 right-3 w-6 h-6 border-b-2 border-r-2 border-primary/60 rounded-br-lg" />
        </div>

        {/* Status badge */}
        <div className="absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-primary/90 backdrop-blur-sm">
          <span className="text-[10px] font-bold text-primary-foreground uppercase tracking-widest">
            Analyzing
          </span>
        </div>
      </div>

      {/* Steps progress */}
      <div className="space-y-2">
        {STEPS.map((step, i) => {
          const isActive = i === currentStep;
          const isDone = i < currentStep;

          return (
            <div
              key={i}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-300 ${
                isActive
                  ? "bg-primary/10 border border-primary/30"
                  : isDone
                  ? "opacity-60"
                  : "opacity-30"
              }`}
            >
              {/* Step indicator */}
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                isDone
                  ? "bg-success text-success-foreground"
                  : isActive
                  ? "bg-primary text-primary-foreground animate-pulse-glow"
                  : "bg-muted text-muted-foreground"
              }`}>
                {isDone ? "✓" : i + 1}
              </div>

              <div className="min-w-0">
                <p className={`text-sm font-semibold truncate ${isActive ? "text-foreground" : "text-muted-foreground"}`}>
                  {step.label}
                </p>
                <p className="text-[10px] text-muted-foreground font-mono truncate">
                  {step.detail}
                </p>
              </div>

              {/* Spinner for active step */}
              {isActive && (
                <div className="ml-auto shrink-0 w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AnalysisAnimation;

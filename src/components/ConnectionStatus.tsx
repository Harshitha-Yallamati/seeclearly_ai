import { useEffect, useState } from "react";

const API_BASE_URL = "http://localhost:5001";

type Status = "checking" | "live" | "fallback" | "disconnected";

const ConnectionStatus = () => {
  const [status, setStatus] = useState<Status>("checking");
  const [mode, setMode] = useState<string>("");

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        signal: AbortSignal.timeout(3000),
      });

      if (!response.ok) {
        setStatus("disconnected");
        setMode("MOCK");
        return;
      }

      const data = await response.json();
      const nextMode = String(data.mode || "LIVE");
      setMode(nextMode);
      setStatus(nextMode.includes("MOCK") ? "fallback" : "live");
    } catch {
      setStatus("disconnected");
      setMode("MOCK");
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const config = {
    checking: {
      color: "bg-yellow-400",
      text: "Checking",
      textColor: "text-yellow-400",
    },
    live: {
      color: "bg-emerald-400",
      text: "Live Model",
      textColor: "text-emerald-400",
    },
    fallback: {
      color: "bg-orange-400",
      text: "Fallback Mode",
      textColor: "text-orange-400",
    },
    disconnected: {
      color: "bg-orange-400",
      text: "Backend Offline",
      textColor: "text-orange-400",
    },
  };

  const current = config[status];

  return (
    <div
      className="flex items-center gap-2 rounded-full bg-muted/50 px-2.5 py-1"
      title={mode || status}
    >
      <div
        className={`h-2 w-2 rounded-full ${current.color} ${
          status === "live" ? "animate-pulse" : ""
        }`}
      />
      <span
        className={`text-[10px] font-semibold uppercase tracking-wider ${current.textColor}`}
      >
        {current.text}
      </span>
    </div>
  );
};

export default ConnectionStatus;

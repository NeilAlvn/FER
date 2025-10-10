// public/js/generateJson.js
export function initGenerateAndSave(videoElement, segmentRef) {
  const generateBtn = document.getElementById("generateJson");
  const saveBtn = document.getElementById("saveEdits");

  if (!generateBtn || !saveBtn) {
    console.warn("Generate/Save buttons not found in DOM.");
    return;
  }

  generateBtn.addEventListener("click", () => {
    const splits = segmentRef.splits || [];
    const video = videoElement;

    if (!video || !video.duration || !video.currentSrc) {
      alert("⚠️ No video loaded.");
      return;
    }

    if (splits.length === 0) {
      alert("⚠️ No segments to generate.");
      return;
    }

    const filename = video.currentSrc.split("/").pop();
    const folderName = video.currentSrc.split("/").slice(-2, -1)[0];

    const data = {
      video: filename,
      folder: folderName,
      totalDuration: video.duration.toFixed(2),
      segments: splits.map((s) => ({
        start: s.start,
        end: s.end,
        emotion: s.emotion || "Neutral",
      })),
    };

    console.log("Generated JSON:", data);
    alert("JSON generated successfully! (Check console)");

    window.latestSegments = data;
  });

  saveBtn.addEventListener("click", async () => {
    if (!window.latestSegments) {
      alert("⚠️ Please generate JSON first.");
      return;
    }

    try {
      const res = await fetch("/api/save-emotions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(window.latestSegments),
      });

      const result = await res.json();
      if (result.ok) {
        alert(`emotions.json saved successfully to exam/${window.latestSegments.folder}/`);
      } else {
        alert("Failed to save emotions.json.");
      }
    } catch (err) {
      console.error("Error saving emotions.json:", err);
      alert("Error saving emotions.json.");
    }
  });
}

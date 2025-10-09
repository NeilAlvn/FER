export function initVideoSegments() {
  console.log("videoSegment.js initialized");

  const video = document.getElementById("videoPlayer");
  const playPause = document.getElementById("playPause");
  const markSplit = document.getElementById("markSplit");
  const undoSplit = document.getElementById("undoSplit");
  const timeline = document.getElementById("videoTimeline");
  const progressBar = document.getElementById("timelineProgress");

  if (!video || !playPause || !markSplit || !undoSplit || !timeline || !progressBar) {
    console.warn("Missing video segment controls.");
    return;
  }

  // Store all split points (in seconds)
  let splits = [];

  /** --- PLAY / PAUSE --- */
  playPause.addEventListener("click", () => {
    if (video.paused) {
      video.play();
      playPause.textContent = "⏸ Pause";
    } else {
      video.pause();
      playPause.textContent = "▶ Play";
    }
  });

  /** --- UPDATE TIMELINE PROGRESS --- */
  video.addEventListener("timeupdate", () => {
    const progress = (video.currentTime / video.duration) * 100;
    progressBar.style.width = `${progress}%`;
  });

  /** --- SPLIT SEGMENT --- */
  markSplit.addEventListener("click", () => {
    if (!video.duration) return;
    const currentTime = video.currentTime;

    splits.push(currentTime);
    addTimelineMarker(currentTime);

    console.log("Added split at", formatTime(currentTime));
  });

  /** --- UNDO LAST SPLIT --- */
  undoSplit.addEventListener("click", () => {
    if (splits.length === 0) return;
    const last = splits.pop();
    removeLastMarker();
    console.log("Removed last split at", formatTime(last));
  });

  /** --- ADD MARKER ON TIMELINE --- */
  function addTimelineMarker(time) {
    const marker = document.createElement("div");
    marker.className = "timeline-marker";
    marker.style.position = "absolute";
    marker.style.top = "0";
    marker.style.width = "3px";
    marker.style.height = "100%";
    marker.style.background = "red";
    marker.style.left = `${(time / video.duration) * 100}%`;
    timeline.appendChild(marker);
  }

  /** --- REMOVE LAST MARKER --- */
  function removeLastMarker() {
    const markers = timeline.querySelectorAll(".timeline-marker");
    if (markers.length > 0) markers[markers.length - 1].remove();
  }

  /** --- HELPER --- */
  function formatTime(seconds) {
    const m = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    const s = Math.floor(seconds % 60)
      .toString()
      .padStart(2, "0");
    return `${m}:${s}`;
  }

  // Debugging helper (optional)
  window.getSplits = () => splits;
}

// public/js/videoSegment.js
export function initVideoSegment(videoElement) {
  const video = document.getElementById("videoPlayer");
  const playPauseBtn = document.getElementById("playPause");
  const splitBtn = document.getElementById("markSplit");
  const undoBtn = document.getElementById("undoSplit");
  const timeline = document.getElementById("videoTimeline");
  const progressBar = document.getElementById("timelineProgress");

  if (!video || !playPauseBtn || !splitBtn || !undoBtn || !timeline) {
    console.warn("Video segment elements missing.");
    return;
  }

  console.log("Video Segment Initialized Successfully");

  let splits = []; // will store { start, end, emotion }
  let isPlaying = false;
  let lastSplit = 0;
  let currentVideoSrc = "";

  // Reset function to clear all segments and markers
  function resetVideoSegments() {
    splits = [];
    lastSplit = 0;
    isPlaying = false;
    
    // Clear timeline markers
    timeline.querySelectorAll(".segment-marker").forEach(el => el.remove());
    
    // Clear split list
    const container = document.getElementById("splitList");
    if (container) {
      container.innerHTML = "";
    }
    
    // Reset play button
    playPauseBtn.textContent = "Play";
    
    // Reset progress bar
    progressBar.style.width = "0%";
    
    console.log("Video segments reset for new video");
  }

  // Detect when video source changes and reset segments
  video.addEventListener("loadstart", () => {
    if (video.src && video.src !== currentVideoSrc) {
      currentVideoSrc = video.src;
      resetVideoSegments();
    }
  });

  // Also reset when src attribute changes (for immediate detection)
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
        if (video.src && video.src !== currentVideoSrc) {
          currentVideoSrc = video.src;
          resetVideoSegments();
        }
      }
    });
  });
  observer.observe(video, { attributes: true, attributeFilter: ['src'] });

  // Make reset function globally available
  window.resetVideoSegments = resetVideoSegments;

  //track video progress
  video.addEventListener("timeupdate", () => {
    if (!video.duration) return;
    const percent = (video.currentTime / video.duration) * 100;
    progressBar.style.width = `${percent}%`;
  });

  //play-pause
  playPauseBtn.addEventListener("click", () => {
    if (isPlaying) {
      video.pause();
      playPauseBtn.textContent = "Play";
    } else {
      video.play();
      playPauseBtn.textContent = "Pause";
    }
    isPlaying = !isPlaying;
  });

  //split segment
  splitBtn.addEventListener("click", () => {
    const currentTime = Number(video.currentTime.toFixed(2));
    const lastTime = Number(lastSplit);

    if (currentTime <= lastTime) {
      alert("Can't Split Before Last Segment!");
      return;
    }

    const segment = {
      start: lastTime,
      end: currentTime,
      emotion: "Neutral"
    };

    splits.push(segment);
    lastSplit = currentTime;

    renderSplits();
    drawTimeLineMarkers();
  });

  //undo last split
  undoBtn.addEventListener("click", () => {
    if (splits.length > 0) {
      const removed = splits.pop();

      if (splits.length > 0) {
        lastSplit = Number(splits[splits.length - 1].end);
      } else {
        lastSplit = 0;
      }
      renderSplits();
      drawTimeLineMarkers();
    }
  });

  function renderSplits() {
    let container = document.getElementById("splitList");
    if (!container) {
      container = document.createElement("div");
      container.id = "splitList";
      container.style.marginTop = "10px";
      video.parentNode.appendChild(container);
    }

    container.innerHTML = "";

    splits.forEach((s, index) => {
      const div = document.createElement("div");
      div.className = "split-item";
      div.style.marginBottom = "5px";
      div.innerHTML = `
        <span>Segment ${index + 1}: ${s.start}s - ${s.end}s</span>
        <select id="emotion-${index}">
          <option value="Surprise" ${s.emotion === "Surprise" ? "selected" : ""}>Surprise</option>
          <option value="Fear" ${s.emotion === "Fear" ? "selected" : ""}>Fear</option>
          <option value="Happiness" ${s.emotion === "Happiness" ? "selected" : ""}>Happiness</option>
          <option value="Sadness" ${s.emotion === "Sadness" ? "selected" : ""}>Sadness</option>
          <option value="Anger" ${s.emotion === "Anger" ? "selected" : ""}>Anger</option>
          <option value="Neutral" ${s.emotion === "Neutral" ? "selected" : ""}>Neutral</option>
        </select>
      `;
      div.querySelector("select").addEventListener("change", (e) => {
        s.emotion = e.target.value;
        drawTimeLineMarkers();
      });
      container.appendChild(div);
    });
  }

  function drawTimeLineMarkers() {
    timeline.querySelectorAll(".segment-marker").forEach(el => el.remove());

    const duration = video.duration || 1;
    const emotionColors = {
      Surprise: "rgba(255, 165, 0, 0.6)",
      Fear: "rgba(138, 43, 226, 0.6)",
      Happiness: "rgba(0, 255, 0, 0.6)",
      Sadness: "rgba(0, 191, 255, 0.6)",
      Anger: "rgba(255, 0, 0, 0.6)",
      Neutral: "rgba(128, 128, 128, 0.5)"
    };

    splits.forEach((s) => {
      const startPercent = (s.start / duration) * 100;
      const endPercent = (s.end / duration) * 100;

      const marker = document.createElement("div");
      marker.className = "segment-marker";
      marker.style.position = "absolute";
      marker.style.left = `${startPercent}%`;
      marker.style.width = `${endPercent - startPercent}%`;
      marker.style.height = "100%";
      marker.style.top = "0";
      marker.style.background = emotionColors[s.emotion] || "rgba(0,0,0,0.3)";
      marker.style.borderRight = "1px solid #000";

      timeline.appendChild(marker);
    });
  }

  // Exported reference for generateJson.js
  window.getVideoSplits = () => splits;
}
function initVideoList() {
  const videosTree = document.getElementById("videosTree");
  const refreshBtn = document.getElementById("refreshVideos");
  const videoPlayer = document.getElementById("videoPlayer");

  if (!videosTree || !refreshBtn) {
    console.warn("Video list not initialized: missing elements.");
    return;
  }

  async function loadVideos() {
    videosTree.innerHTML = "Loading videos...";
    try {
      const res = await fetch("/api/videos");
      const videos = await res.json();

      videosTree.innerHTML = "";
      videos.forEach((video) => {
        const a = document.createElement("a");
        a.href = "#";
        a.textContent = video;
        a.style.display = "block";

        a.addEventListener("click", () => {
          document.querySelectorAll("#videosTree a").forEach((el) => el.classList.remove("active"));
          a.classList.add("active");
          videoPlayer.src = `/uploads/${video}`;
          
          // Reset video segments when a new video is loaded
          if (window.resetVideoSegments) {
            window.resetVideoSegments();
          }
        });

        videosTree.appendChild(a);
      });
    } catch (err) {
      console.error("Failed to load videos:", err);
      videosTree.innerHTML = "Error loading videos.";
    }
  }

  refreshBtn.addEventListener("click", loadVideos);
  loadVideos();
}
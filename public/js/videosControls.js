console.log("videoControls initialized");

export function initVideoControls() {
  const videoListContainer = document.getElementById("videoList");
  const videoPlayer = document.getElementById("videoPlayer");

  if (!videoListContainer) {
    console.warn("videoList element not found in admin.html");
    return;
  }
  if (!videoPlayer) {
    console.warn("videoPlayer element not found in admin.html");
    return;
  }

  fetch("/api/videos")
    .then(res => res.json())
    .then(data => {
      videoListContainer.innerHTML = "";

      Object.keys(data).forEach(username => {
        const userSection = document.createElement("div");
        userSection.classList.add("user-section");

        const userHeader = document.createElement("h3");
        userHeader.textContent = username;
        userHeader.classList.add("username");
        userHeader.style.cursor = "pointer";

        // List of that user's videos (initially hidden)
        const list = document.createElement("ul");
        list.style.display = "none";

        data[username].forEach(video => {
          const item = document.createElement("li");
          const btn = document.createElement("button");
          btn.textContent = video;
          btn.classList.add("video-btn");
          btn.addEventListener("click", () => {
            const videoPath = `/videos/${username}/${video}`;
            videoPlayer.src = videoPath;
            videoPlayer.play();
            
            // Reset video segments when a new video is loaded
            if (window.resetVideoSegments) {
              window.resetVideoSegments();
            }
          });
          item.appendChild(btn);
          list.appendChild(item);
        });

        // Click username to toggle its video list
        userHeader.addEventListener("click", () => {
          // Hide all other lists first
          document.querySelectorAll("#videoList ul").forEach(ul => {
            if (ul !== list) ul.style.display = "none";
          });

          // Toggle this one
          list.style.display = list.style.display === "none" ? "block" : "none";
        });

        userSection.appendChild(userHeader);
        userSection.appendChild(list);
        videoListContainer.appendChild(userSection);
      });
    })
    .catch(err => {
      console.error("Error fetching videos:", err);
      videoListContainer.innerHTML = "<p>Failed to load videos.</p>";
    });
}
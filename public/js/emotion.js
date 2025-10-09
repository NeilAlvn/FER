export function initEmotions() {
    console.log("emotions.js initialized");
    const emotionList = document.getElementById("emotionList");
    const emotionsUl = document.getElementById("emotions");

    if (!emotionList || !emotionsUl) {
        console.warn("Emotions module not initialized: missing elements.");
        return;
  }

  // Temporary right-click handler
  document.addEventListener("contextmenu", (e) => {
    if (e.target.tagName === "A" && e.target.href.includes(".mp4")) {
      e.preventDefault();
      emotionList.style.display = "block";
      emotionList.style.top = e.pageY + "px";
      emotionList.style.left = e.pageX + "px";
      loadEmotions(); // Mocked
    }
  });

  function loadEmotions() {
    emotionsUl.innerHTML = "";
    const mockEmotions = ["Happy", "Sad", "Neutral", "Surprised"];
    mockEmotions.forEach((emo) => {
      const li = document.createElement("li");
      li.textContent = emo;
      emotionsUl.appendChild(li);
    });
  }
}

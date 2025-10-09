function initStorage() {
  const totalSecondsInput = document.getElementById("totalSeconds");
  const saveAllBtn = document.getElementById("saveAll");
  const generateJsonBtn = document.getElementById("generateJson");
  const saveEditsBtn = document.getElementById("saveEdits");

  if (!saveAllBtn || !generateJsonBtn) {
    console.warn("Storage module not initialized: missing elements.");
    return;
  }

  saveAllBtn.addEventListener("click", () => {
    const total = parseInt(totalSecondsInput.value);
    if (!total || total <= 0) {
      alert("Please enter a valid total seconds value.");
      return;
    }

    const data = {
      totalSeconds: total,
      questions: window.getQuestions?.() || [],
      splits: window.getVideoSplits?.() || [],
    };

    console.log("Saving all data:", data);
    alert("Data saved successfully (console only for now).");
  });

  generateJsonBtn.addEventListener("click", () => {
    const data = {
      questions: window.getQuestions?.() || [],
      splits: window.getVideoSplits?.() || [],
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "exam_data.json";
    a.click();
    URL.revokeObjectURL(url);
  });

  saveEditsBtn.addEventListener("click", () => {
    alert("Edits saved (placeholder).");
  });
}

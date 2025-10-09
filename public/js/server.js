// public/js/server.js

export async function saveQuestionsToServer(payload) {
  try {
    const res = await fetch("/api/questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await res.json();
    console.log("Server response:", result);
    return result.ok;
  } catch (err) {
    console.error("Save error:", err);
    return false;
  }
}

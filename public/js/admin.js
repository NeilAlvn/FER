import { initVideoControls } from './videosControls.js';
import { initEmotions } from './emotion.js';
import { initQuestionEditor } from './questionEditor.js';
import { initVideoSegment } from './videoSegment.js';
import { initGenerateAndSave } from './generateJson.js';

document.addEventListener('DOMContentLoaded', () => {
  console.log("admin.js loaded successfully");
  initVideoControls();
  initEmotions();
  initQuestionEditor();
  initVideoSegment();

  initGenerateAndSave();
});

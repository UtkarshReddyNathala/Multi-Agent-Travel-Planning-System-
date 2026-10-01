// TripMate AI front end: sends trip requests to the API, shows the draft plan,
// handles human approval, and renders the execution dashboard.

// The thread ID is kept in localStorage so follow-up messages continue the same conversation.
let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let waitingForApproval = false;

const AGENT_LABELS = {
  flight_agent: "✈️ Flight Agent",
  hotel_agent: "🏨 Hotel Agent",
  weather_agent: "🌦️ Weather Agent",
  budget_agent: "💰 Budget Agent",
  itinerary_agent: "🗓️ Itinerary Agent"
};

// ---------------------------------------------------------------------------
// UI helpers: example prompts, loading state, errors and Markdown rendering
// ---------------------------------------------------------------------------
function setPrompt(text) {
  document.getElementById("userInput").value = text;
}

function setLoading(isLoading, mode = "draft") {
  const sendBtn = document.getElementById("sendBtn");
  const btnText = document.getElementById("btnText");
  const btnLoader = document.getElementById("btnLoader");
  const approveBtn = document.getElementById("approveBtn");
  const reviseBtn = document.getElementById("reviseBtn");

  sendBtn.disabled = isLoading;
  approveBtn.disabled = isLoading;
  reviseBtn.disabled = isLoading;

  if (isLoading && mode === "draft") {
    btnText.classList.add("hidden");
    btnLoader.classList.remove("hidden");
  } else {
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
  }
}

function showError(message) {
  const errorBox = document.getElementById("errorBox");
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
  const errorBox = document.getElementById("errorBox");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

// The plan is LLM output that can include text from web search results, so the
// HTML produced by marked is sanitised with DOMPurify before it is shown.
// If either library failed to load, the plan is shown as plain text instead.
function renderMarkdown(element, markdown) {
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    element.innerHTML = DOMPurify.sanitize(marked.parse(markdown || ""));
  } else {
    element.innerText = markdown || "";
  }
}

// Show which agents the supervisor chose for this request.
function showWorkflow(data) {
  const section = document.getElementById("workflowSection");
  const reasoning = document.getElementById("supervisorReasoning");
  const chips = document.getElementById("agentChips");
  const guardrailBadge = document.getElementById("guardrailBadge");

  reasoning.textContent = data.supervisor_reasoning || "Supervisor routing completed.";
  chips.innerHTML = "";

  (data.selected_agents || []).forEach((agent) => {
    const chip = document.createElement("span");
    chip.className = "agent-chip";
    chip.textContent = AGENT_LABELS[agent] || agent;
    chips.appendChild(chip);
  });

  if (data.guardrail_allowed === false) {
    guardrailBadge.textContent = "Guardrail blocked";
    guardrailBadge.classList.add("blocked");
  } else {
    guardrailBadge.textContent = "Guardrail passed";
    guardrailBadge.classList.remove("blocked");
  }

  section.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Agent Execution Dashboard: shows `data.dashboard` exactly as the backend
// sent it (measured timings and real statuses). Text is added with
// textContent, so error messages can never inject HTML.
// ---------------------------------------------------------------------------
const STAGE_ICONS = {
  input_validation: "🧹",
  guardrail: "🛡️",
  supervisor: "🧭",
  flight_agent: "✈️",
  hotel_agent: "🏨",
  weather_agent: "🌦️",
  budget_agent: "💰",
  itinerary_agent: "🗓️",
  human_approval: "👤",
  final_agent: "✅"
};

const STATUS_LABELS = {
  success: "Success",
  degraded: "Partial / fallback",
  failed: "Failed",
  blocked: "Blocked",
  not_run: "Not run",
  not_selected: "Not selected",
  pending: "Pending",
  awaiting: "Awaiting review",
  approved: "Approved",
  revision_requested: "Revision requested"
};

const WORKFLOW_LABELS = {
  awaiting_approval: "Awaiting human approval",
  completed: "Completed",
  blocked: "Blocked by guardrail",
  failed: "Failed",
  in_progress: "In progress"
};

const CACHE_LABELS = {
  hit: "Redis cache hit",
  miss: "Redis cache miss",
  unavailable: "Redis unavailable, live data used",
  disabled: "Cache disabled"
};

function addSummaryItem(container, label, value) {
  const item = document.createElement("div");
  item.className = "summary-item";
  const l = document.createElement("span");
  l.className = "summary-label";
  l.textContent = label;
  const v = document.createElement("strong");
  v.textContent = value;
  item.appendChild(l);
  item.appendChild(v);
  container.appendChild(item);
}

function showDashboard(data) {
  const dashboard = data.dashboard;
  const section = document.getElementById("dashboardSection");
  if (!dashboard || !Array.isArray(dashboard.stages)) {
    section.classList.add("hidden");
    return;
  }

  const badge = document.getElementById("workflowStatusBadge");
  badge.textContent = WORKFLOW_LABELS[dashboard.workflow_status] || dashboard.workflow_status;
  badge.className = "guardrail-badge";
  if (dashboard.workflow_status === "blocked" || dashboard.workflow_status === "failed") {
    badge.classList.add("blocked");
  } else if (dashboard.has_warnings) {
    badge.classList.add("warning");
  }

  const summary = document.getElementById("dashboardSummary");
  summary.innerHTML = "";
  addSummaryItem(summary, "Measured time", `${(dashboard.measured_ms / 1000).toFixed(1)} s`);
  addSummaryItem(summary, "LLM calls", String(dashboard.llm_calls));
  addSummaryItem(summary, "Warnings", dashboard.has_warnings ? "Yes, see stages below" : "None");

  const list = document.getElementById("dashboardStages");
  list.innerHTML = "";

  dashboard.stages.forEach((stage) => {
    const li = document.createElement("li");
    li.className = `stage stage-${stage.status}`;

    const icon = document.createElement("span");
    icon.className = "stage-icon";
    icon.textContent = STAGE_ICONS[stage.key] || "•";

    const body = document.createElement("div");
    body.className = "stage-body";

    const title = document.createElement("div");
    title.className = "stage-title";
    const name = document.createElement("span");
    name.textContent = stage.label;
    const pill = document.createElement("span");
    pill.className = "stage-pill";
    pill.textContent = STATUS_LABELS[stage.status] || stage.status;
    title.appendChild(name);
    title.appendChild(pill);
    body.appendChild(title);

    const meta = [];
    if (typeof stage.duration_ms === "number") {
      meta.push(stage.duration_ms >= 1000
        ? `${(stage.duration_ms / 1000).toFixed(1)} s`
        : `${Math.round(stage.duration_ms)} ms`);
    }
    if (stage.cache) {
      meta.push(CACHE_LABELS[stage.cache] || stage.cache);
    }
    if (meta.length) {
      const metaLine = document.createElement("div");
      metaLine.className = "stage-meta";
      metaLine.textContent = meta.join(" · ");
      body.appendChild(metaLine);
    }

    if (stage.detail) {
      const detail = document.createElement("div");
      detail.className = "stage-detail";
      detail.textContent = stage.detail;
      body.appendChild(detail);
    }
    if (stage.error) {
      const error = document.createElement("div");
      error.className = "stage-error";
      error.textContent = stage.error;
      body.appendChild(error);
    }

    li.appendChild(icon);
    li.appendChild(body);
    list.appendChild(li);
  });

  section.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Conversation: show the current thread and start a new one
// ---------------------------------------------------------------------------
function updateConversationInfo() {
  const info = document.getElementById("conversationInfo");
  if (!info) return;
  info.textContent = currentThreadId
    ? "Continuing your current conversation. Follow-up messages reuse the trip details."
    : "New conversation. Follow-up messages (for example “change the budget to 25,000”) reuse this trip.";
}

function newConversation() {
  currentThreadId = null;
  localStorage.removeItem("travel_thread_id");
  latestAnswerMarkdown = "";
  hideError();
  hideApproval();
  ["workflowSection", "dashboardSection", "resultSection"].forEach((id) => {
    document.getElementById(id).classList.add("hidden");
  });
  document.getElementById("userInput").value = "";
  updateConversationInfo();
}

// ---------------------------------------------------------------------------
// Results and human approval (approve the draft or ask for changes)
// ---------------------------------------------------------------------------
function showResult(answer, threadId, isDraft = false) {
  latestAnswerMarkdown = answer || "";

  const resultSection = document.getElementById("resultSection");
  const resultBox = document.getElementById("resultBox");
  const threadInfo = document.getElementById("threadInfo");
  const resultTitle = document.getElementById("resultTitle");

  renderMarkdown(resultBox, latestAnswerMarkdown);
  threadInfo.textContent = `Thread ID: ${threadId}`;
  resultTitle.textContent = isDraft ? "Draft Travel Plan" : "Your Final AI Travel Plan";
  resultSection.classList.remove("hidden");

  resultSection.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}

function showApproval(data) {
  waitingForApproval = true;
  const section = document.getElementById("approvalSection");
  const approvalRequest = document.getElementById("approvalRequest");
  approvalRequest.textContent = data.approval_request ||
    "Approve the draft or provide feedback before the final plan is generated.";
  section.classList.remove("hidden");
}

function hideApproval() {
  waitingForApproval = false;
  document.getElementById("approvalSection").classList.add("hidden");
  document.getElementById("approvalFeedback").value = "";
}

// Send the user's message to POST /api/travel and show the draft for approval.
async function sendMessage() {
  hideError();

  if (waitingForApproval) {
    showError("Please approve or revise the current draft before starting another plan.");
    return;
  }

  const input = document.getElementById("userInput");
  const message = input.value.trim();

  if (!message) {
    showError("Please enter your travel request first.");
    return;
  }

  setLoading(true, "draft");

  try {
    const response = await fetch("/api/travel", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: message,
        thread_id: currentThreadId
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Something went wrong.");
    }

    currentThreadId = data.thread_id;
    localStorage.setItem("travel_thread_id", currentThreadId);

    showWorkflow(data);
    showDashboard(data);
    updateConversationInfo();

    if (data.requires_approval) {
      showResult(data.itinerary || data.answer, data.thread_id, true);
      showApproval(data);
    } else {
      hideApproval();
      showResult(data.answer, data.thread_id, false);
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "draft");
  }
}

// Send the approval decision (and feedback, if rejected) to POST /api/travel/approve.
async function submitApproval(approved) {
  hideError();

  if (!currentThreadId || !waitingForApproval) {
    showError("There is no draft waiting for approval.");
    return;
  }

  const feedbackInput = document.getElementById("approvalFeedback");
  const feedback = feedbackInput.value.trim();

  if (!approved && !feedback) {
    showError("Please enter revision feedback before requesting changes.");
    feedbackInput.focus();
    return;
  }

  setLoading(true, "approval");

  try {
    const response = await fetch("/api/travel/approve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        thread_id: currentThreadId,
        approved: approved,
        feedback: feedback
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Could not resume the travel workflow.");
    }

    showWorkflow(data);
    showDashboard(data);
    hideApproval();
    showResult(data.answer, data.thread_id, false);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "approval");
  }
}

// ---------------------------------------------------------------------------
// Export: copy the plan or download it as a PDF
// ---------------------------------------------------------------------------
function copyResult() {
  const resultBox = document.getElementById("resultBox");
  const text = resultBox.innerText;

  if (!text) {
    return;
  }

  navigator.clipboard.writeText(text)
    .then(() => {
      const copyBtn = document.querySelector(".copy-btn");
      const oldText = copyBtn.textContent;
      copyBtn.textContent = "Copied!";

      setTimeout(() => {
        copyBtn.textContent = oldText;
      }, 1400);
    })
    .catch(() => {
      showError("Could not copy result.");
    });
}

function downloadPDF() {
  const pdfContent = document.getElementById("pdfContent");

  if (!latestAnswerMarkdown || !pdfContent) {
    showError("No travel plan available to download.");
    return;
  }

  const downloadBtn = document.querySelector(".download-btn");
  const oldText = downloadBtn.textContent;
  downloadBtn.textContent = "Preparing PDF...";
  downloadBtn.disabled = true;

  const options = {
    margin: 0.5,
    filename: "ai-travel-plan.pdf",
    image: {
      type: "jpeg",
      quality: 0.98
    },
    html2canvas: {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff"
    },
    jsPDF: {
      unit: "in",
      format: "a4",
      orientation: "portrait"
    },
    pagebreak: {
      mode: ["avoid-all", "css", "legacy"]
    }
  };

  html2pdf()
    .set(options)
    .from(pdfContent)
    .save()
    .then(() => {
      downloadBtn.textContent = oldText;
      downloadBtn.disabled = false;
    })
    .catch(() => {
      downloadBtn.textContent = oldText;
      downloadBtn.disabled = false;
      showError("Could not download PDF.");
    });
}

// Press Ctrl+Enter to send the message.
document.addEventListener("keydown", function(event) {
  if (event.ctrlKey && event.key === "Enter") {
    sendMessage();
  }
});

updateConversationInfo();

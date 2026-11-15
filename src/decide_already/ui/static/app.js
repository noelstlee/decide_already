"use strict";

// The board page of decide_already. Plain JavaScript: no libraries and no build step.
//
// One loop runs everything. An action calls the server through `api`, `refresh` then asks
// for the board again, and `render` redraws the page from `state`. Nothing is drawn from
// anywhere else. Text from the vault (file names, descriptions, card text) is not trusted:
// it only ever enters the page as text, through `el`.

const state = {
  board: null, // the last answer of GET /api/board
  loadError: null, // why the board could not be loaded, when it could not
  selected: null, // id of the file that is open in the side panel
  detail: null, // the last answer of GET /api/files/<selected>
  form: null, // the open group form: {id, name, covers, excludes, confirmDelete}
  sort: null, // the last answer of GET /api/sort/status
  watching: false, // true while this page follows a running sort
  adding: 0, // how many dropped files are still on their way to the dump folder
  // What went wrong and stays on the page until dismissed: {title, rows: [[file, reason]]}.
  // `sort` is what the last sort could not do, `drop` the dropped files that were not added.
  problems: { sort: null, drop: null },
  dragging: null, // id of the file whose card is being dragged
  moving: null, // id of a file whose move the server has not answered yet
  pressing: false, // true while a mouse button or a finger is down
  redrawDue: false, // true when a redraw was put off until the press or the drag is over
  ticket: 0, // counts refreshes, so that only the newest answer is drawn
};

const UNSORTED = "Unsorted"; // the name of the lane for files that are in no group
const inFlight = new Set(); // the names of the requests that are not answered yet (see once)

// The element with this id. The ids are the ones in index.html.
const $ = (id) => document.getElementById(id);

// Small helpers

// Build an element. Strings among the children become text, never HTML. An attribute that is
// null, undefined or false is left out, and an attribute named on<event> is a listener.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

// Join class names, leaving out the ones that are switched off.
const cls = (...names) => names.filter(Boolean).join(" ");

// A button with a label and what pressing it does.
function button(label, onclick, attrs = {}) {
  return el("button", { type: "button", class: "button", ...attrs, onclick }, label);
}

// A count with its noun: "1 file", "3 files".
const plural = (count, noun) => `${count} ${noun}${count === 1 ? "" : "s"}`;

// A file name with a line break allowed after each "_", "-" and ".", so that a long name wraps
// between its parts and not in the middle of a word.
function breakable(name) {
  const parts = name.match(/[^_.-]*[_.-]?/g).filter(Boolean);
  return parts.flatMap((part, index) => (index === 0 ? [part] : [el("wbr"), part]));
}

// A probability with two decimals, or "?" when the model gave none.
const prob = (value) => (typeof value === "number" ? value.toFixed(2) : "?");

// Dollars: four decimals for the small amounts a sort costs, two from one cent up.
function money(value) {
  if (value === 0) return "$0.00";
  return `$${value.toFixed(value < 0.01 ? 4 : 2)}`;
}

// A file id as a URL path. Each part is escaped on its own, so the slashes stay.
const filePath = (fileId) => fileId.split("/").map(encodeURIComponent).join("/");

// The folder part of a file id, with its last slash: "hw3/" for "hw3/solution.py".
const folderOf = (file) => file.file_id.slice(0, file.file_id.length - file.name.length);

// The server

// The one way to talk to the server. Gives back the parsed JSON, or throws an Error whose
// message is the reason: the server's own "detail" line when it sent one.
async function api(method, path, body) {
  const options = { method };
  if (body instanceof Blob) {
    options.body = body; // a dropped file goes as it is
  } else if (body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("the server is not answering (is p0 ui still running?)");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : "";
    throw new Error(detail || `the server answered with status ${response.status}`);
  }
  return data;
}

// Messages

// Show a short message that goes away by itself. A click takes it away sooner.
function toast(message, kind = "info") {
  const box = $("toasts");
  const note = el("div", { class: cls("toast", kind), onclick: () => note.remove() }, message);
  box.append(note);
  while (box.children.length > 4) box.firstElementChild.remove();
  setTimeout(() => note.remove(), kind === "error" ? 10000 : 5000);
}

// Run one action. When it fails, say on the page what did not happen, and why. Gives back
// whether it worked.
async function act(lead, work) {
  try {
    await work();
    return true;
  } catch (error) {
    toast(`${lead}: ${error.message}`, "error");
    return false;
  }
}

// Run `work` unless the request of this name is still in flight: a second press on a button
// whose first press the server has not answered yet does nothing.
async function once(name, work) {
  if (inFlight.has(name)) return;
  inFlight.add(name);
  try {
    await work();
  } finally {
    inFlight.delete(name);
  }
}

// "3 files were not sorted", "1 file was not added": the headline of a list of problems.
function notDone(count, what) {
  return `${plural(count, "file")} ${count === 1 ? "was" : "were"} not ${what}`;
}

// Looking things up on the board

// True while a sort runs. Nothing may change the vault meanwhile.
const sorting = () => Boolean(state.sort && state.sort.running);

// Every file on the board, Unsorted first.
function allFiles() {
  if (!state.board) return [];
  return [...state.board.unsorted, ...state.board.groups.flatMap((column) => column.files)];
}

// The board's entry for a file id. Undefined for a file that is not on the board.
const findFile = (fileId) => allFiles().find((file) => file.file_id === fileId);

// A group's column on the board: {group, files}. Undefined for an id that is no group now.
const findColumn = (groupId) => state.board.groups.find((entry) => entry.group.id === groupId);

// The name to show for a group id: Unsorted for null, the id itself for a group that is gone.
function groupName(groupId) {
  if (groupId === null) return UNSORTED;
  const column = findColumn(groupId);
  return column ? column.group.name : groupId;
}

// The group the model gave the highest probability. Ties go to the lower id, as in the sorter.
function topGroup(file) {
  const ranked = Object.entries(file.probs || {}).sort(
    ([idA, probA], [idB, probB]) => probB - probA || (idA < idB ? -1 : 1),
  );
  return ranked.length ? ranked[0][0] : null;
}

// Loading

// Ask the server for the board, and for the open file, then redraw. When several of these
// run at once, only the newest answer is drawn. When the open file can no longer be shown
// (its original left the dump folder, say), the side panel closes and a message says why.
async function refresh() {
  const ticket = ++state.ticket;
  const selected = state.selected;
  const board = await api("GET", "/api/board");
  let detail = null;
  let gone = "it is no longer on the board";
  if (selected) {
    detail = await api("GET", `/api/files/${filePath(selected)}`).catch((error) => {
      gone = error.message;
      return null;
    });
  }
  if (ticket !== state.ticket) return;
  state.board = board;
  state.loadError = null;
  if (state.selected === selected) {
    state.detail = detail;
    if (selected && (!detail || !findFile(selected))) {
      state.selected = null;
      toast(`The panel of ${selected} was closed: ${gone}`, "error");
    }
  }
  if (board.sorting && !sorting()) state.sort = { running: true, done: 0, total: 0 };
  render();
  if (board.sorting) watchSort();
}

// Load the board: for the first time, after an action, or when coming back to the page.
// This never throws. The reason for a failure is shown in place of the board when there is
// no board yet, and as a message when the board on the page could not be brought up to date.
async function load() {
  try {
    await refresh();
  } catch (error) {
    state.loadError = error.message;
    if (state.board) toast(`Board not refreshed: ${error.message}`, "error");
    render();
  }
}

// Coming back to the page (from Obsidian, say) picks up what changed in the vault meanwhile.
function refreshOnReturn() {
  if (state.dragging || state.watching) return;
  load();
}

// Drawing

// Redraw the whole page from `state`. Focus, the text cursor and every scroll position are
// put back afterwards, so a redraw is not felt.
//
// A redraw replaces every element. That would lose a click whose mouse button is still down
// and break a drag, so while a press or a drag is under way the redraw waits (see catchUp).
function render() {
  if (state.pressing || state.dragging) {
    state.redrawDue = true;
    return;
  }
  state.redrawDue = false;
  const focus = focusMark();
  const scroll = scrollMarks();
  renderHeader();
  renderBoard();
  renderPanel();
  restoreScroll(scroll);
  restoreFocus(focus);
}

// Do the redraw that was put off, if one was.
function catchUp() {
  if (state.redrawDue) render();
}

// What has the focus: its data-key, and the text cursor when it is a text field.
function focusMark() {
  const node = document.activeElement;
  const key = node && node.dataset ? node.dataset.key : null;
  if (!key) return null;
  const typing = node.tagName === "INPUT" || node.tagName === "TEXTAREA";
  return { key, cursor: typing ? [node.selectionStart, node.selectionEnd] : null };
}

// Put the focus, and the text cursor, back where focusMark found them.
function restoreFocus(mark) {
  if (!mark) return;
  const node = focusKey(mark.key);
  if (node && mark.cursor) node.setSelectionRange(mark.cursor[0], mark.cursor[1]);
}

// Give the focus to the element with this data-key, if it is on the page.
function focusKey(key) {
  const node = document.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (node) node.focus({ preventScroll: true });
  return node;
}

// Where every scrolling part of the page stands, by its data-scroll name.
function scrollMarks() {
  const marks = new Map();
  for (const node of document.querySelectorAll("[data-scroll]")) {
    marks.set(node.dataset.scroll, [node.scrollLeft, node.scrollTop]);
  }
  return marks;
}

// Put every scrolling part back where scrollMarks found it.
function restoreScroll(marks) {
  for (const node of document.querySelectorAll("[data-scroll]")) {
    const mark = marks.get(node.dataset.scroll);
    if (mark) [node.scrollLeft, node.scrollTop] = mark;
  }
}

// The header: the vault's name, set version, model and spend, then the buttons.
function renderHeader() {
  const board = state.board;
  const vault = board ? board.vault : null;
  document.title = vault ? `${vault.name} board` : "decide_already";
  $("vault-name").textContent = vault ? vault.name : "decide_already";
  $("vault-name").title = vault ? vault.path : "";
  $("meta").hidden = !vault;
  if (vault) {
    $("meta-set").textContent = board.set_version;
    $("meta-model").textContent = vault.model;
    $("meta-spend").textContent = `${money(vault.spent_usd)} of ${money(vault.max_spend_usd)}`;
  }
  const waiting = board ? board.new_files.length : 0;
  const adding = state.adding > 0;
  $("pending").hidden = waiting === 0 && !adding;
  $("pending").textContent = adding
    ? `Adding ${plural(state.adding, "file")}`
    : plural(waiting, "new file");
  $("add-group").disabled = !board || sorting();
  $("add-files").disabled = !board; // uploads are fine during a sort: it does not see them
  renderSortButton();
}

// The Sort button. While a sort runs it shows how far the sort is and cannot be pressed.
// It cannot be pressed while dropped files are still arriving either: a sort started then
// would not see them.
function renderSortButton() {
  const { done = 0, total = 0 } = state.sort || {};
  const running = sorting();
  let label = "Sort";
  let blocked = "";
  if (running) label = total ? `Sorting ${done} of ${total}` : "Sorting";
  else if (!state.board) blocked = "The board is not loaded.";
  else if (state.adding > 0) blocked = "Files are still being added.";
  else if (state.board.groups.length === 0) blocked = "Add a group first.";
  else if (allFiles().length + state.board.new_files.length === 0) blocked = "Add files first.";
  $("sort-label").textContent = label;
  $("sort-fill").style.width = running && total ? `${(100 * done) / total}%` : "0";
  $("sort").disabled = running || blocked !== "";
  $("sort").title = blocked || (running ? "" : "Sorts every file that is not frozen");
  if (running) $("sort").setAttribute("aria-busy", "true");
  else $("sort").removeAttribute("aria-busy");
}

// The board: the Unsorted lane, one lane per group, and the form for a new group.
function renderBoard() {
  const root = $("board");
  const board = state.board;
  if (!board) {
    const waitingNote = el("p", { class: "quiet-note" }, "Loading");
    root.replaceChildren(state.loadError ? loadFailed() : waitingNote);
    return;
  }
  const lanes = [];
  const showUnsorted = board.groups.length > 0 || board.unsorted.length > 0;
  if (showUnsorted) lanes.push(unsortedLane(board.unsorted));
  for (const column of board.groups) lanes.push(groupLane(column));
  if (state.form && state.form.id === null) {
    lanes.push(el("section", { class: "lane", "aria-label": "New group" }, groupForm()));
  }
  const problems = ["sort", "drop"].map(problemList).filter(Boolean);
  if (lanes.length === 0) {
    root.replaceChildren(...problems, firstGroupInvite());
    return;
  }
  const notice = boardNotice();
  root.replaceChildren(
    ...problems,
    ...(notice ? [el("p", { class: "notice" }, notice)] : []),
    el("div", { class: "lanes", "data-scroll": "lanes" }, lanes),
  );
}

// A list of things that went wrong, above the lanes: a headline, then every file with its
// reason. Unlike a message it stays until it is dismissed (or, for a sort, until the next
// sort). `kind` is its name in state.problems. Null when there is nothing to list.
function problemList(kind) {
  const list = state.problems[kind];
  if (!list) return null;
  const dismiss = () => {
    state.problems[kind] = null;
    render();
    focusKey("sort");
  };
  const rows = list.rows.map(([file, reason]) =>
    el("li", {}, file && el("span", { class: "problem-file" }, `${file}: `), reason),
  );
  return el(
    "section",
    { class: "problems", "aria-label": list.title },
    el(
      "div",
      { class: "problems-head" },
      el("h2", {}, list.title),
      button("Dismiss", dismiss, { class: "button quiet", "data-key": `dismiss:${kind}` }),
    ),
    el("ul", { class: "problems-list", "data-scroll": `problems:${kind}` }, rows),
  );
}

// "3 files are waiting in the dump folder", for files that were never sorted.
function waitingFiles(count) {
  return `${plural(count, "file")} ${count === 1 ? "is" : "are"} waiting in the dump folder`;
}

// What a new vault shows: an invitation to add the first group.
function firstGroupInvite() {
  const waiting = state.board.new_files.length;
  const next = waiting ? `${waitingFiles(waiting)}.` : "Then drop files anywhere on this page.";
  return el(
    "div",
    { class: "invite" },
    el("h2", {}, "Start with a group"),
    el(
      "p",
      {},
      "A group is a name and a short description of what belongs in it, like Homework or " +
        "Lecture notes. The model sorts every file into one of your groups.",
    ),
    button("Add your first group", () => openForm(null), {
      class: "button primary",
      "data-key": "first-group",
    }),
    el("p", {}, next),
  );
}

// What is shown when the board cannot be loaded: the reason, and a way to try again.
function loadFailed() {
  return el(
    "div",
    { class: "invite" },
    el("h2", {}, "The board could not be loaded"),
    el("p", {}, `Reason: ${state.loadError}`),
    button("Try again", load, { class: "button primary", "data-key": "retry" }),
  );
}

// One line of guidance above the lanes, for a board that has nothing to show yet.
function boardNotice() {
  const board = state.board;
  const waiting = board.new_files.length;
  if (board.groups.length === 0) {
    return state.form ? null : "There are no groups. Add a group, then sort.";
  }
  if (allFiles().length > 0 || sorting()) return null;
  if (waiting === 0) {
    return (
      "No files yet. Drop files anywhere on this page, or copy them into the " +
      `${board.vault.dump_dir} folder of the vault.`
    );
  }
  const them = waiting === 1 ? "it" : "them";
  return `${waitingFiles(waiting)}. Press Sort to place ${them} in your groups.`;
}

// The Unsorted lane: files that no group fits well enough, or that you put there.
function unsortedLane(files) {
  const head = el(
    "header",
    { class: "lane-head" },
    laneTitle(UNSORTED, files.length),
    el("p", { class: "lane-covers" }, "Files that no group fits well enough."),
    laneActions(null, UNSORTED, files),
  );
  return lane(null, UNSORTED, "lane unsorted", head, files, "Nothing is unsorted.");
}

// A group's lane. Its head shows the group, or the form while the group is being edited.
function groupLane({ group, files }) {
  const editing = state.form && state.form.id === group.id;
  const head = editing
    ? groupForm()
    : el(
        "header",
        { class: "lane-head" },
        laneTitle(group.name, files.length),
        el("p", { class: "lane-covers" }, group.covers),
        group.excludes && el("p", { class: "lane-excludes" }, `Excludes: ${group.excludes}`),
        laneActions(group, group.name, files),
      );
  return lane(group.id, group.name, "lane", head, files, "No files here yet.");
}

// One lane: its head, then its cards. data-group is where a dropped card is moved to, and
// it is empty for Unsorted.
function lane(groupId, name, className, head, files, emptyText) {
  const key = groupId === null ? "" : groupId;
  const label = `${name}, ${plural(files.length, "file")}`;
  const cards = files.length ? files.map(card) : el("p", { class: "empty" }, emptyText);
  return el(
    "section",
    { class: className, "data-group": key, "aria-label": label },
    head,
    el("div", { class: "cards", "data-scroll": `lane:${key}` }, cards),
  );
}

// A lane's name with the number of files in it.
function laneTitle(name, count) {
  return el(
    "div",
    { class: "lane-title" },
    el("h2", {}, name),
    el("span", { class: "count", title: plural(count, "file") }, String(count)),
  );
}

// The quiet buttons under a lane's description: edit the group, freeze or unfreeze its files.
function laneActions(group, name, files) {
  const groupId = group ? group.id : null; // null is the Unsorted lane, as in a move
  const ref = group ? group.id : UNSORTED;
  const allFrozen = files.length > 0 && files.every((file) => file.frozen);
  const quiet = (key) => ({ class: "button quiet", disabled: sorting(), "data-key": key });
  return el(
    "div",
    { class: "lane-actions" },
    group && button("Edit", () => openForm(group), quiet(`edit:${ref}`)),
    files.length > 0 &&
      button(
        allFrozen ? "Unfreeze all" : "Freeze all",
        () => freezeLane(groupId, name, !allFrozen),
        quiet(`freeze:${ref}`),
      ),
  );
}

// One file's card. A click or Enter opens it in the side panel, and it can be dragged to
// another lane.
function card(file) {
  const open = () => openFile(file.file_id);
  const reason = file.group_id === null ? unsortedReason(file) : null;
  return el(
    "div",
    {
      class: cls(
        "card",
        file.file_id === state.selected && "selected",
        file.file_id === state.moving && "lifted",
      ),
      role: "button",
      tabindex: "0",
      "aria-current": file.file_id === state.selected && "true",
      draggable: sorting() ? null : "true",
      title: file.file_id,
      "data-file": file.file_id,
      "data-key": `card:${file.file_id}`,
      onclick: open,
      onkeydown: (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        open();
      },
    },
    el("span", { class: "card-name" }, cardFolder(file), breakable(file.name)),
    el("span", { class: "card-type" }, file.file_type),
    cardProbability(file),
    reason && el("span", { class: "card-reason" }, reason),
    marks(file),
  );
}

// The folder a file sits in, shown before its name in a quieter color.
function cardFolder(file) {
  const folder = folderOf(file);
  return folder && el("span", { class: "card-folder" }, folder);
}

// The model's top probability as a thin bar with its number. When the top group is not the
// lane the card sits in (it is in Unsorted, or you moved it), the group is named.
function cardProbability(file) {
  if (file.top_prob === null) return null;
  const top = topGroup(file);
  const elsewhere = top !== null && top !== file.group_id;
  const number = prob(file.top_prob);
  return el(
    "span",
    { class: "card-prob" },
    bar(file.top_prob),
    el("span", { class: "sr" }, "Top probability"),
    el("span", { class: "num" }, elsewhere ? `${groupName(top)} ${number}` : number),
  );
}

// A probability drawn as a length.
function bar(value) {
  const fill = el("span", { class: "bar-fill" });
  fill.style.width = `${Math.round(value * 100)}%`;
  return el("span", { class: "bar", "aria-hidden": "true" }, fill);
}

// Why a file is in Unsorted, in a few plain words.
function unsortedReason(file) {
  if (file.reason === "no_group_fits") return "No group fits";
  if (file.reason === "weak_top") return "No group stands out";
  if (file.reason === "no_text") return "No text to read";
  if (file.placed_by === "code") return "Its group was deleted";
  return null; // you put it here, and the mark says so
}

// The small marks on a card: close call, frozen, moved by you.
function marks(file) {
  const labels = [
    file.close_call && el("span", { class: "mark accent" }, "Close call"),
    file.frozen && el("span", { class: "mark" }, "Frozen"),
    file.placed_by === "you" && el("span", { class: "mark" }, "Moved by you"),
  ].filter(Boolean);
  return labels.length ? el("span", { class: "marks" }, labels) : null;
}

// The group form

// The form in a lane's head: a new group, or an existing one edited in place. What is typed
// is kept in state.form, so a redraw does not lose it.
function groupForm() {
  const form = state.form;
  const field = (name, label, hint, control) => {
    control.value = form[name];
    control.addEventListener("input", () => {
      form[name] = control.value;
    });
    return el(
      "label",
      { class: "field" },
      el("span", { class: "field-label" }, label),
      control,
      el("span", { class: "field-hint" }, hint),
    );
  };
  const nameHint = "The model reads the name too, so say what the group holds.";
  return el(
    "form",
    { class: "group-form", novalidate: true, onsubmit: submitGroup, onkeydown: formKeys },
    field("name", "Name", nameHint, el("input", { type: "text", "data-key": "form:name" })),
    field(
      "covers",
      "Covers",
      "What belongs in this group.",
      el("textarea", { rows: "3", "data-key": "form:covers" }),
    ),
    field(
      "excludes",
      "Excludes",
      "Optional. What does not belong.",
      el("textarea", { rows: "2", "data-key": "form:excludes" }),
    ),
    form.confirmDelete ? deleteConfirm() : formButtons(),
  );
}

// The buttons under the form: save, cancel and, for an existing group, delete.
function formButtons() {
  const isNew = state.form.id === null;
  const save = { type: "submit", class: "button primary", "data-key": "form:save" };
  const remove = { class: "button quiet", "data-key": "form:delete" };
  return el(
    "div",
    { class: "form-actions" },
    el("button", save, isNew ? "Add group" : "Save"),
    button("Cancel", closeForm, { "data-key": "form:cancel" }),
    !isNew && button("Delete group", askDelete, remove),
  );
}

// Deleting asks first, in the form itself.
function deleteConfirm() {
  const column = findColumn(state.form.id);
  const count = column ? column.files.length : 0;
  const name = column ? column.group.name : "this group";
  const question = count
    ? `Delete ${name}? Its ${plural(count, "file")} will go to Unsorted.`
    : `Delete ${name}? It holds no files.`;
  return el(
    "div",
    { class: "confirm" },
    el("p", {}, question),
    el(
      "div",
      { class: "form-actions" },
      button("Delete group", deleteGroup, { class: "button primary", "data-key": "form:confirm" }),
      button("Keep group", keepGroup, { "data-key": "form:keep" }),
    ),
  );
}

// Keys inside the form: Escape leaves it, and Ctrl or Cmd with Enter saves it.
function formKeys(event) {
  if (event.key === "Escape") {
    event.stopPropagation();
    closeForm();
  } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    saveGroup();
  }
}

// The form was submitted (Enter in the name field, or the save button).
function submitGroup(event) {
  event.preventDefault();
  saveGroup();
}

// Open the form for a group, or for a new group when none is given. A new group's form that
// is already open keeps what was typed in it.
function openForm(group) {
  if (!group && state.form && state.form.id === null) {
    focusKey("form:name");
    return;
  }
  state.form = { id: null, name: "", covers: "", excludes: "", confirmDelete: false };
  if (group) {
    const { id, name, covers, excludes } = group;
    state.form = { id, name, covers, excludes: excludes || "", confirmDelete: false };
  }
  render();
  const input = focusKey("form:name");
  // The whole lane is brought into view, not just the input: a new group's form is the last
  // lane, and the input alone would leave the lane's edge cut off.
  if (input) {
    (input.closest(".lane") || input).scrollIntoView({ block: "nearest", inline: "nearest" });
  }
}

// Leave the form without saving, and give the focus back to the button that opened it.
function closeForm() {
  const id = state.form ? state.form.id : null;
  state.form = null;
  render();
  focusKey(id === null ? "add-group" : `edit:${id}`);
}

// Show the question that comes before a delete.
function askDelete() {
  state.form.confirmDelete = true;
  render();
  focusKey("form:keep");
}

// The answer was no: back to the form's own buttons.
function keepGroup() {
  state.form.confirmDelete = false;
  render();
  focusKey("form:delete");
}

// Save the form: add the new group, or send what changed in an existing one. A second press
// while the first is not answered does nothing. When the server refuses, the board is loaded
// again, which also shows a sort that is running (the usual reason for a refusal).
async function saveGroup() {
  const form = state.form;
  if (!form) return; // pressed again after the save went through, before the form was redrawn
  const name = form.name.trim();
  const covers = form.covers.trim();
  const excludes = form.excludes.trim();
  if (!name || !covers) {
    toast("A group needs a name and a covers text.", "error");
    focusKey(name ? "form:covers" : "form:name");
    return;
  }
  await once("group", async () => {
    const saved = await act("Not saved", async () => {
      const id =
        form.id === null ? await addGroup(name, covers, excludes) : await changeGroup(form);
      state.form = null;
      await load();
      focusKey(`edit:${id}`);
    });
    if (!saved) await load();
  });
}

// Add a group and give back its id.
async function addGroup(name, covers, excludes) {
  const body = { name, covers, excludes: excludes || null };
  const answer = await api("POST", "/api/groups", body);
  toast(`Added ${name}. Sort to place files in it.`);
  return answer.group.id;
}

// Send what the form changed in a group, when it changed anything, and give back its id.
async function changeGroup(form) {
  const column = findColumn(form.id);
  if (!column) throw new Error("this group is no longer there");
  const group = column.group;
  const changes = {};
  for (const field of ["name", "covers", "excludes"]) {
    const typed = form[field].trim();
    if (typed !== (group[field] || "")) changes[field] = typed;
  }
  if (Object.keys(changes).length > 0) {
    await api("PATCH", `/api/groups/${encodeURIComponent(form.id)}`, changes);
    toast(`Saved ${form.name.trim()}. Sort again to apply the change.`);
  }
  return form.id;
}

// The answer was yes: delete the group. Its files go to Unsorted. A second press and a
// refusal by the server are handled as in saveGroup.
async function deleteGroup() {
  if (!state.form) return;
  const id = state.form.id;
  const name = groupName(id);
  await once("group", async () => {
    const deleted = await act("Not deleted", async () => {
      await api("DELETE", `/api/groups/${encodeURIComponent(id)}`);
      state.form = null;
      toast(`Deleted ${name}. Any files it held are in Unsorted now.`);
      await load();
      focusKey("add-group");
    });
    if (!deleted) await load();
  });
}

// The side panel

// The side panel: everything about the selected file. Hidden when no file is selected.
function renderPanel() {
  const panel = $("panel");
  const file = state.selected ? findFile(state.selected) : null;
  if (!file || !state.detail) {
    panel.hidden = true;
    panel.replaceChildren();
    return;
  }
  panel.hidden = false;
  panel.replaceChildren(
    panelHead(file),
    panelStatus(file),
    panelActions(file),
    panelSaid(file, state.detail),
    panelSaw(state.detail),
  );
}

// The top of the panel: the file's name, its type and folder, and the way out.
function panelHead(file) {
  const folder = folderOf(file);
  const kind = folder ? `${file.file_type}, in ${folder}` : file.file_type;
  const title = el("h2", { class: "panel-title" }, breakable(file.name));
  return el(
    "header",
    { class: "panel-head" },
    el("div", {}, title, el("p", { class: "panel-sub" }, kind)),
    button("Close", closePanel, { class: "button quiet", "data-key": "panel:close" }),
  );
}

// Where the file is and how it got there, in plain words.
function panelStatus(file) {
  const where = groupName(file.group_id);
  let placed = `In Unsorted. ${unsortedWhy(file)}`;
  if (file.placed_by === "you") placed = `You placed this in ${where}.`;
  else if (file.group_id !== null) placed = `The model placed this in ${where}.`;
  const frozen = file.frozen
    ? "Frozen: sorting leaves it where it is."
    : "Not frozen: the next sort decides again.";
  return el(
    "div",
    { class: "panel-status" },
    el("p", {}, placed),
    el("p", { class: "frozen-line" }, frozen),
  );
}

// The longer reason a file is in Unsorted, with the numbers and the cutoffs behind it.
function unsortedWhy(file) {
  const cutoffs = state.board.vault.cutoffs;
  const under = (cutoff) => `under the cutoff of ${prob(cutoff)}`;
  if (file.reason === "no_group_fits") {
    return `No group fits: "fits any group" is ${prob(file.fits_any)}, ${under(cutoffs.fits_any)}.`;
  }
  if (file.reason === "weak_top") {
    const top = prob(file.top_prob);
    return `No group stands out: the top probability is ${top}, ${under(cutoffs.top)}.`;
  }
  if (file.reason === "no_text") return "No text could be read, so the model was not asked.";
  if (file.placed_by === "code") return "Its group was deleted.";
  return "";
}

// Move to another group, freeze or unfreeze, open the original.
function panelActions(file) {
  const busy = sorting();
  const current = file.group_id === null ? "" : file.group_id;
  const choices = state.board.groups.map(({ group }) => [group.id, group.name]);
  choices.push(["", UNSORTED]);
  const select = el(
    "select",
    { id: "move-to", disabled: busy, "data-key": "panel:move-to" },
    choices.map(([value, label]) => el("option", { value, selected: value === current }, label)),
  );
  const move = () => {
    const target = select.value === "" ? null : select.value;
    if (target === file.group_id) toast(`${file.name} is already in ${groupName(target)}.`);
    else moveFile(file.file_id, target);
  };
  const moveTitle = "Moves the file and freezes it there";
  return el(
    "div",
    { class: "panel-actions" },
    el(
      "div",
      { class: "move" },
      el("label", { for: "move-to" }, "Move to"),
      select,
      button("Move", move, { disabled: busy, title: moveTitle, "data-key": "panel:move" }),
    ),
    button(file.frozen ? "Unfreeze" : "Freeze", () => freezeFile(file, !file.frozen), {
      disabled: busy,
      "data-key": "panel:freeze",
    }),
    el(
      "a",
      {
        class: "button",
        href: `/api/original/${filePath(file.file_id)}`,
        target: "_blank",
        rel: "noopener",
        "data-key": "panel:original",
      },
      "Open original",
    ),
  );
}

// What the model said: every group's probability as a bar, then "fits any group".
function panelSaid(file, detail) {
  const placement = detail.placement;
  const section = el("section", { class: "panel-section" }, el("h3", {}, "What the model said"));
  if (detail.probs.length === 0) {
    section.append(el("p", { class: "explain" }, "The model was not asked about this file."));
    return section;
  }
  const rows = detail.probs.map(({ group_id: groupId, name, prob: value }) =>
    probRow(name === null ? `${groupId} (deleted)` : name, value, groupId === file.group_id),
  );
  section.append(el("ul", { class: "probs" }, rows));
  if (placement.fits_any !== null) {
    const fits = probRow("Fits any group", placement.fits_any, false);
    section.append(el("ul", { class: "probs fits" }, fits));
  }
  const notes = [`Answered by ${placement.model} under set version ${placement.set_version}.`];
  if (file.stale) notes.push("The groups have changed since then.");
  if (file.close_call) {
    const margin = prob(state.board.vault.cutoffs.close_call);
    notes.push(`Close call: the top two groups are less than ${margin} apart.`);
  }
  section.append(el("p", { class: "explain" }, notes.join(" ")));
  return section;
}

// One row of the probability list: a name, a bar, a number.
function probRow(name, value, current) {
  return el(
    "li",
    { class: cls("prob", current && "current") },
    el("span", { class: "prob-name", title: name }, name),
    bar(value),
    el("span", { class: "num" }, prob(value)),
  );
}

// What the model saw: the card, section by section, as the card note has it.
function panelSaw(detail) {
  const section = el("section", { class: "panel-section" }, el("h3", {}, "What the model saw"));
  const title = cardTitle(detail.card);
  const parts = cardSections(detail.card);
  if (title) section.append(el("p", { class: "saw-title" }, `Title: ${title}`));
  for (const part of parts) {
    section.append(el("h4", {}, part.heading), el("pre", { class: "card-text" }, part.text));
  }
  if (parts.length === 0 && detail.placement.reason === "no_text") {
    section.append(el("p", { class: "explain" }, "Nothing. No text could be read from this file."));
  } else if (parts.length === 0) {
    // A note in another shape than card.py writes is shown as it is.
    section.append(el("pre", { class: "card-text" }, detail.card.trim()));
  }
  return section;
}

// The sections of a card note's body: [{heading, text}]. The body is markdown: "## Heading"
// lines, each followed by its text inside a fence of backticks.
function cardSections(body) {
  const lines = body.split("\n");
  const sections = [];
  let index = 0;
  while (index < lines.length) {
    const heading = /^## (.+)$/.exec(lines[index]);
    index += 1;
    if (!heading) continue;
    while (index < lines.length && lines[index] === "") index += 1;
    const fence = /^(`{3,})text$/.exec(lines[index] || "");
    if (!fence) continue;
    const end = lines.indexOf(fence[1], index + 1);
    if (end === -1) break;
    sections.push({ heading: heading[1], text: lines.slice(index + 1, end).join("\n") });
    index = end + 1;
  }
  return sections;
}

// The title line of a card note, without the code span it is wrapped in.
function cardTitle(body) {
  const intro = body.split("\n## ")[0];
  const line = /^- Title: (.*)$/m.exec(intro);
  if (!line) return null;
  const span = /^(`+) ?(.*?) ?\1$/.exec(line[1]);
  return span ? span[2] : line[1];
}

// Actions on files

// Open a file in the side panel. A second press on the open file's card closes it.
async function openFile(fileId) {
  if (state.selected === fileId) {
    closePanel();
    return;
  }
  await act("Not opened", async () => {
    state.detail = await api("GET", `/api/files/${filePath(fileId)}`);
    state.selected = fileId;
    render();
    $("panel").scrollTop = 0;
    $("panel").focus({ preventScroll: true });
  });
}

// Close the side panel, and give the focus back to the file's card.
function closePanel() {
  const fileId = state.selected;
  state.selected = null;
  state.detail = null;
  render();
  if (fileId) focusKey(`card:${fileId}`);
}

// Put a file in a group, or in Unsorted for null. The server freezes it there. Until the
// server has answered, the card stays where it was, dimmed.
async function moveFile(fileId, groupId) {
  const file = findFile(fileId);
  if (!file || file.group_id === groupId) return;
  state.moving = fileId;
  render();
  await act("Not moved", async () => {
    await api("POST", "/api/move", { file_id: fileId, group_id: groupId });
    toast(`Moved ${file.name} to ${groupName(groupId)} and froze it there.`);
  });
  state.moving = null;
  await load();
}

// Freeze or unfreeze one file. The request says that it names a file, so a file that is
// called like a group is never taken for the group.
async function freezeFile(file, frozen) {
  await act(frozen ? "Not frozen" : "Not unfrozen", async () => {
    await api("POST", frozen ? "/api/freeze" : "/api/unfreeze", { file_id: file.file_id });
    const then = frozen ? "Sorting leaves it where it is." : "The next sort decides again.";
    toast(`${frozen ? "Froze" : "Unfroze"} ${file.name}. ${then}`);
  });
  await load();
}

// Freeze or unfreeze every file in a lane: a group's by its id, the Unsorted lane by null.
async function freezeLane(groupId, name, frozen) {
  await act(frozen ? "Not frozen" : "Not unfrozen", async () => {
    const body = { group_id: groupId };
    const answer = await api("POST", frozen ? "/api/freeze" : "/api/unfreeze", body);
    const count = (frozen ? answer.frozen : answer.unfrozen).length;
    toast(`${frozen ? "Froze" : "Unfroze"} ${plural(count, "file")} in ${name}.`);
  });
  await load();
}

// Sorting

// Start a sort on the server, then follow it. A second press while the first is not
// answered does nothing. When the server refuses, the board is loaded again: if the reason
// is a sort that is already running (started in another tab, say), the page then follows it.
async function startSort() {
  if (sorting() || state.adding > 0) return; // the button may still show while a redraw waits
  await once("sort", async () => {
    const started = await act("Sort not started", async () => {
      state.sort = await api("POST", "/api/sort");
    });
    if (!started) {
      await load();
      return;
    }
    watchSort();
    render();
  });
}

// Follow a running sort: show how far it is, let cards land as they are placed, and say how
// it ended. What the sort before it could not do is taken off the page.
async function watchSort() {
  if (state.watching) return;
  state.watching = true;
  state.problems.sort = null;
  let drawn = -1;
  try {
    for (;;) {
      state.sort = await api("GET", "/api/sort/status");
      if (!state.sort.running) break;
      if (state.sort.done === drawn) {
        renderHeader();
      } else {
        drawn = state.sort.done;
        await refresh();
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    // The sort is over. How it ended is said even when the board cannot be loaded just now:
    // what stopped the sort (a group note that cannot be read, say) often stops the board too.
    await load();
    reportSort(state.sort);
  } catch (error) {
    state.sort = null;
    render();
    toast(`Lost track of the sort: ${error.message}`, "error");
  } finally {
    state.watching = false;
  }
}

// Say how a sort ended: what it placed, and what it could not sort. A message goes away by
// itself, so the reason a sort stopped and every file that was not sorted are also put in a
// list that stays above the board (see problemList).
function reportSort(status) {
  if (status.error) {
    state.problems.sort = { title: "The sort stopped", rows: [[null, status.error]] };
    toast(`The sort stopped: ${status.error}`, "error");
    render();
    return;
  }
  const report = status.report;
  if (!report) return;
  const parts = [`${report.placed} placed`, `${report.unsorted} in Unsorted`];
  if (report.close_calls) parts.push(plural(report.close_calls, "close call"));
  if (report.frozen_skipped) parts.push(`${report.frozen_skipped} frozen left alone`);
  toast(`Sorted: ${parts.join(", ")}.`);
  const failed = Object.entries(report.errors);
  if (failed.length > 0) {
    state.problems.sort = { title: notDone(failed.length, "sorted"), rows: failed };
    toast(`${notDone(failed.length, "sorted")}. The list is above the board.`, "error");
    render();
  }
}

// Dragging cards between lanes

let dropLane = null; // the lane a dragged card is over right now

// Mark the lane a dragged card is over, and take the mark off the lane it was over before.
function markDropLane(node) {
  if (node === dropLane) return;
  if (dropLane) dropLane.classList.remove("drop");
  if (node) node.classList.add("drop");
  dropLane = node;
}

// The group id a lane stands for. Null for Unsorted.
const groupOfLane = (node) => (node.dataset.group === "" ? null : node.dataset.group);

// The lane a dragged card may be dropped on: any lane but the one it is already in.
// (Some browsers give a text node as the target of a drag event.)
function dropTarget(event) {
  const under = event.target instanceof Element ? event.target : event.target.parentElement;
  const node = under ? under.closest("[data-group]") : null;
  const file = findFile(state.dragging);
  if (!node || !file) return null;
  return groupOfLane(node) === file.group_id ? null : node;
}

// A drag is over, dropped or not. The mouse button is up too, whatever the browser reported.
// The redraw takes the dimming off the dragged card and does any redraw that was put off.
function endDrag() {
  state.dragging = null;
  state.pressing = false;
  markDropLane(null);
  render();
}

// Listen for cards being dragged. A drop on another lane moves the file there.
function watchCardDrags() {
  document.addEventListener("dragstart", (event) => {
    const dragged = event.target instanceof Element ? event.target.closest(".card") : null;
    if (!dragged) return;
    state.dragging = dragged.dataset.file;
    event.dataTransfer.setData("text/plain", dragged.dataset.file);
    event.dataTransfer.effectAllowed = "move";
    // The card is dimmed a moment later, so that the picture that follows the mouse is not.
    // This is done on the card itself: a redraw would replace it and end the drag.
    setTimeout(() => dragged.classList.add("lifted"), 0);
  });
  document.addEventListener("dragover", (event) => {
    if (!state.dragging) return;
    const target = dropTarget(event);
    markDropLane(target);
    if (!target) return;
    event.preventDefault(); // this is what allows a drop here
    event.dataTransfer.dropEffect = "move";
  });
  document.addEventListener("drop", (event) => {
    if (!state.dragging) return;
    const target = dropTarget(event);
    if (!target) return;
    event.preventDefault();
    const fileId = state.dragging;
    const groupId = groupOfLane(target);
    endDrag();
    moveFile(fileId, groupId);
  });
  document.addEventListener("dragend", endDrag);
}

// Dropping files from the desktop

// Listen for files dragged in from the desktop: show the drop message, then add them.
function watchFileDrops() {
  const overlay = $("drop-overlay");
  const hide = () => {
    overlay.hidden = true;
  };
  const carriesFiles = (event) =>
    Boolean(event.dataTransfer) && Array.from(event.dataTransfer.types).includes("Files");
  // The overlay covers the whole page once files come in, so leaving it means leaving the page.
  window.addEventListener("dragenter", (event) => {
    if (carriesFiles(event)) overlay.hidden = false;
  });
  // Without these two, a file dropped on the page would make the browser open it.
  window.addEventListener("dragover", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("drop", (event) => {
    if (carriesFiles(event)) event.preventDefault();
  });
  overlay.addEventListener("dragleave", hide);
  overlay.addEventListener("click", hide);
  overlay.addEventListener("drop", (event) => {
    hide();
    addFiles(droppedFiles(event.dataTransfer));
  });
}

// Listen for the Add files button: it opens the system's file picker, and the files picked
// are added the same way as dropped ones. The picker offers files only, never folders.
function watchFilePicker() {
  const picker = $("file-picker");
  $("add-files").addEventListener("click", () => picker.click());
  picker.addEventListener("change", () => {
    const files = Array.from(picker.files || []);
    picker.value = ""; // so that picking the same file again still counts as a change
    if (files.length > 0) addFiles({ files, folders: 0 });
  });
}

// The files in a drop. They are read at once: the browser takes a drop's content away again
// when the event is over. Folders cannot be sent as files, so they are only counted.
function droppedFiles(transfer) {
  const files = [];
  let folders = 0;
  for (const item of Array.from(transfer.items)) {
    if (item.kind !== "file") continue;
    const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
    if (entry && entry.isDirectory) folders += 1;
    else files.push(item.getAsFile());
  }
  return { files: files.filter(Boolean), folders };
}

// Send dropped files to the dump folder one by one, then say how it went. Sort cannot be
// pressed meanwhile (see renderSortButton). Every file that was not added is put in a list
// that stays above the board, with its reason (see problemList).
async function addFiles({ files, folders }) {
  const failed = [];
  let added = 0;
  if (state.adding === 0) state.problems.drop = null; // a new drop starts a new list
  state.adding += files.length;
  render();
  for (const file of files) {
    try {
      await api("PUT", `/api/dump/${encodeURIComponent(file.name)}`, file);
      added += 1;
    } catch (error) {
      failed.push([file.name, error.message]);
    }
    state.adding -= 1;
    renderHeader();
  }
  if (added > 0) {
    const them = added === 1 ? "it" : "them";
    toast(`Added ${plural(added, "file")} to the dump folder. Press Sort to place ${them}.`);
  }
  if (folders > 0) toast("Folders are not added. Drop the files inside them.", "error");
  if (failed.length > 0) {
    // Two drops can overlap, so the list is added to and not replaced.
    const rows = [...(state.problems.drop ? state.problems.drop.rows : []), ...failed];
    state.problems.drop = { title: notDone(rows.length, "added"), rows };
    toast(`${notDone(failed.length, "added")}. The list is above the board.`, "error");
  }
  await load();
}

// Starting up

// Keep track of a mouse button or a finger being down, for `render`.
function watchPresses() {
  const release = () => {
    state.pressing = false;
    setTimeout(catchUp, 0); // the click comes right after the release and is handled first
  };
  document.addEventListener("pointerdown", () => {
    state.pressing = true;
  });
  document.addEventListener("pointerup", release);
  document.addEventListener("pointercancel", release);
  // A release that never arrived (the button came up outside the window) must not hold up
  // redraws for good: the next move of the mouse with no button down counts as one.
  document.addEventListener("pointermove", (event) => {
    if (state.pressing && event.buttons === 0) release();
  });
}

// Escape takes away the drop message, or else closes the side panel.
function onEscape(event) {
  if (event.key !== "Escape") return;
  if (!$("drop-overlay").hidden) $("drop-overlay").hidden = true;
  else if (state.selected) closePanel();
}

// Wire up the parts of the page that never change, then load the board.
function start() {
  $("sort").addEventListener("click", startSort);
  $("add-group").addEventListener("click", () => openForm(null));
  document.addEventListener("keydown", onEscape);
  window.addEventListener("focus", refreshOnReturn);
  watchPresses();
  watchCardDrags();
  watchFileDrops();
  watchFilePicker();
  render();
  load();
}

start();

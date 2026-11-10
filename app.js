"use strict";

const MODEL_FILE_PATH = "solace_model.json";
const POST_SEPARATOR_PATTERN = /^\s*---\s*$/m;
const NUMBER_OF_KEYWORDS_PER_CATEGORY = 5;
const KEYWORD_MINIMUM_LENGTH = 3;
const IGNORED_KEYWORDS = new Set(["url"]);
const SENTIMENT_DISPLAY_ORDER = ["negative", "neutral", "positive"];

const TOPIC_DISPLAY_NAMES = {
  events_achievements: "Events & Achievements",
  admissions_greetings: "Admissions & Greetings",
  faculty_admin: "Faculty & Administration",
  facilities_services: "Facilities & Campus Services",
  reviews_questions: "School Reviews & Enrollment Questions",
  other: "Other",
};

const postInputElement = document.getElementById("post-input");
const csvFileInputElement = document.getElementById("csv-file-input");
const csvHasHeaderCheckboxElement = document.getElementById("csv-has-header-checkbox");
const analyzeButtonElement = document.getElementById("analyze-button");
const statusMessageElement = document.getElementById("status-message");
const resultsSectionElement = document.getElementById("results-section");
const overallSummaryElement = document.getElementById("overall-summary");
const categoryListElement = document.getElementById("category-list");

let preparedTopicModel = null;
let preparedSentimentModel = null;


/* ---------- 1. Text cleaning and model math ---------- */

function cleanPostText(rawPostText) {
  return String(rawPostText)
    .normalize("NFKC")
    .replace(/https?:\/\/\S+/g, " url ")
    .replace(/#/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenizeText(cleanedText) {
  return cleanedText.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
}

function prepareModel(exportedModel) {
  return {
    classNames: exportedModel.class_names,
    vocabularyMap: new Map(Object.entries(exportedModel.vocabulary)),
    inverseDocumentFrequency: exportedModel.inverse_document_frequency,
    coefficients: exportedModel.coefficients,
    intercepts: exportedModel.intercepts,
  };
}

function buildTfidfEntries(cleanedText, preparedModel) {
  const tokens = tokenizeText(cleanedText);
  const termCounts = new Map();

  function countTermIfKnown(term) {
    if (preparedModel.vocabularyMap.has(term)) {
      termCounts.set(term, (termCounts.get(term) || 0) + 1);
    }
  }

  tokens.forEach(countTermIfKnown);
  for (let tokenPosition = 0; tokenPosition < tokens.length - 1; tokenPosition++) {
    countTermIfKnown(tokens[tokenPosition] + " " + tokens[tokenPosition + 1]);
  }

  const tfidfEntries = [];
  let sumOfSquares = 0;
  for (const [term, count] of termCounts) {
    const featureIndex = preparedModel.vocabularyMap.get(term);
    const weightedValue = (1 + Math.log(count)) * preparedModel.inverseDocumentFrequency[featureIndex];
    tfidfEntries.push({ term, featureIndex, value: weightedValue });
    sumOfSquares += weightedValue * weightedValue;
  }

  const vectorLength = Math.sqrt(sumOfSquares);
  if (vectorLength > 0) {
    tfidfEntries.forEach((entry) => { entry.value /= vectorLength; });
  }
  return tfidfEntries;
}

function predictClass(cleanedText, preparedModel) {
  const tfidfEntries = buildTfidfEntries(cleanedText, preparedModel);

  const classScores = preparedModel.intercepts.slice();
  for (let classIndex = 0; classIndex < classScores.length; classIndex++) {
    for (const entry of tfidfEntries) {
      classScores[classIndex] += preparedModel.coefficients[classIndex][entry.featureIndex] * entry.value;
    }
  }

  const highestScore = Math.max(...classScores);
  const exponentials = classScores.map((score) => Math.exp(score - highestScore));
  const exponentialsTotal = exponentials.reduce((runningTotal, value) => runningTotal + value, 0);
  const probabilities = exponentials.map((value) => value / exponentialsTotal);

  const bestClassIndex = probabilities.indexOf(Math.max(...probabilities));
  return {
    predictedClassName: preparedModel.classNames[bestClassIndex],
    confidence: probabilities[bestClassIndex],
    tfidfEntries,
  };
}


/* ---------- 2. Batch analysis ---------- */

function extractTopKeywords(postsInCategory) {
  const summedTermScores = new Map();
  for (const analyzedPost of postsInCategory) {
    for (const entry of analyzedPost.topicTfidfEntries) {
      const isSingleWord = !entry.term.includes(" ");
      if (!isSingleWord || entry.term.length < KEYWORD_MINIMUM_LENGTH || IGNORED_KEYWORDS.has(entry.term)) {
        continue;
      }
      summedTermScores.set(entry.term, (summedTermScores.get(entry.term) || 0) + entry.value);
    }
  }
  return [...summedTermScores.entries()]
    .sort((firstEntry, secondEntry) => secondEntry[1] - firstEntry[1])
    .slice(0, NUMBER_OF_KEYWORDS_PER_CATEGORY)
    .map(([term]) => term);
}

function analyzePostBatch(listOfPosts) {
  const analyzedPosts = listOfPosts.map((originalText) => {
    const cleanedText = cleanPostText(originalText);
    const topicPrediction = predictClass(cleanedText, preparedTopicModel);
    const sentimentPrediction = predictClass(cleanedText, preparedSentimentModel);
    return {
      originalText,
      predictedTopic: topicPrediction.predictedClassName,
      predictedSentiment: sentimentPrediction.predictedClassName,
      sentimentConfidence: sentimentPrediction.confidence,
      topicTfidfEntries: topicPrediction.tfidfEntries,
    };
  });

  const postsGroupedByTopic = new Map();
  for (const analyzedPost of analyzedPosts) {
    if (!postsGroupedByTopic.has(analyzedPost.predictedTopic)) {
      postsGroupedByTopic.set(analyzedPost.predictedTopic, []);
    }
    postsGroupedByTopic.get(analyzedPost.predictedTopic).push(analyzedPost);
  }

  const categoryReports = [...postsGroupedByTopic.entries()].map(([topicName, postsInCategory]) => {
    const sentimentCounts = { negative: 0, neutral: 0, positive: 0 };
    postsInCategory.forEach((post) => { sentimentCounts[post.predictedSentiment] += 1; });

    const postsNegativeFirst = [...postsInCategory].sort((firstPost, secondPost) =>
      SENTIMENT_DISPLAY_ORDER.indexOf(firstPost.predictedSentiment) -
      SENTIMENT_DISPLAY_ORDER.indexOf(secondPost.predictedSentiment));

    return {
      topicName,
      numberOfPosts: postsInCategory.length,
      sentimentCounts,
      keywords: extractTopKeywords(postsInCategory),
      posts: postsNegativeFirst,
    };
  });

  categoryReports.sort((firstReport, secondReport) => secondReport.numberOfPosts - firstReport.numberOfPosts);

  const overallSentimentCounts = { negative: 0, neutral: 0, positive: 0 };
  analyzedPosts.forEach((post) => { overallSentimentCounts[post.predictedSentiment] += 1; });

  return { totalNumberOfPosts: analyzedPosts.length, overallSentimentCounts, categoryReports };
}


/* ---------- 3. Rendering ---------- */

function createElementWithText(tagName, textContent, className) {
  const element = document.createElement(tagName);
  element.textContent = textContent;
  if (className) element.className = className;
  return element;
}

function renderSentimentCounts(sentimentCounts) {
  const countsElement = document.createElement("div");
  countsElement.className = "sentiment-counts";
  SENTIMENT_DISPLAY_ORDER.forEach((sentimentName) => {
    countsElement.appendChild(
      createElementWithText("span", `${sentimentName}: ${sentimentCounts[sentimentName]}`, sentimentName));
  });
  return countsElement;
}

function renderSentimentBar(sentimentCounts, numberOfPosts) {
  const barElement = document.createElement("div");
  barElement.className = "sentiment-bar";
  SENTIMENT_DISPLAY_ORDER.forEach((sentimentName) => {
    const count = sentimentCounts[sentimentName];
    if (count === 0) return;
    const segmentElement = document.createElement("div");
    segmentElement.className = `bar-segment ${sentimentName}`;
    segmentElement.style.width = `${(count / numberOfPosts) * 100}%`;
    segmentElement.title = `${sentimentName}: ${count}`;
    barElement.appendChild(segmentElement);
  });
  return barElement;
}

function renderCategoryCard(categoryReport) {
  const cardElement = document.createElement("div");
  cardElement.className = "category-card";

  const displayName = TOPIC_DISPLAY_NAMES[categoryReport.topicName] || categoryReport.topicName;
  cardElement.appendChild(createElementWithText("h3", `${displayName} (${categoryReport.numberOfPosts})`));
  cardElement.appendChild(createElementWithText(
    "p", `Common words: ${categoryReport.keywords.join(", ") || "none"}`, "category-meta"));
  cardElement.appendChild(renderSentimentBar(categoryReport.sentimentCounts, categoryReport.numberOfPosts));
  cardElement.appendChild(renderSentimentCounts(categoryReport.sentimentCounts));

  const detailsElement = document.createElement("details");
  detailsElement.appendChild(createElementWithText("summary", `Show posts (${categoryReport.numberOfPosts})`));

  const postListElement = document.createElement("ul");
  postListElement.className = "post-list";
  categoryReport.posts.forEach((analyzedPost) => {
    const listItemElement = document.createElement("li");
    const confidencePercent = Math.round(analyzedPost.sentimentConfidence * 100);
    listItemElement.appendChild(createElementWithText(
      "span", `${analyzedPost.predictedSentiment} ${confidencePercent}%`,
      `sentiment-tag ${analyzedPost.predictedSentiment}`));
    listItemElement.appendChild(createElementWithText("p", analyzedPost.originalText, "post-text"));
    postListElement.appendChild(listItemElement);
  });
  detailsElement.appendChild(postListElement);
  cardElement.appendChild(detailsElement);

  return cardElement;
}

function renderResults(batchAnalysis) {
  const { totalNumberOfPosts, overallSentimentCounts, categoryReports } = batchAnalysis;
  overallSummaryElement.textContent =
    `${totalNumberOfPosts} posts in ${categoryReports.length} categories. ` +
    `Negative: ${overallSentimentCounts.negative}, ` +
    `Neutral: ${overallSentimentCounts.neutral}, ` +
    `Positive: ${overallSentimentCounts.positive}.`;

  categoryListElement.replaceChildren(...categoryReports.map(renderCategoryCard));
  resultsSectionElement.hidden = false;
}


/* ---------- 4. Input handling ---------- */

function splitIntoPosts(rawInputText, splitMode) {
  const pieces = splitMode === "separator"
    ? rawInputText.split(POST_SEPARATOR_PATTERN)
    : rawInputText.split(/\r?\n/);
  return pieces.map((piece) => piece.trim()).filter((piece) => piece.length > 0);
}

function parseCsvIntoRows(csvText) {
  const rows = [];
  let currentRow = [];
  let currentField = "";
  let insideQuotes = false;

  for (let position = 0; position < csvText.length; position++) {
    const character = csvText[position];
    if (insideQuotes) {
      if (character === '"') {
        if (csvText[position + 1] === '"') { currentField += '"'; position++; }
        else { insideQuotes = false; }
      } else {
        currentField += character;
      }
    } else if (character === '"') {
      insideQuotes = true;
    } else if (character === ",") {
      currentRow.push(currentField);
      currentField = "";
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && csvText[position + 1] === "\n") position++;
      currentRow.push(currentField);
      rows.push(currentRow);
      currentRow = [];
      currentField = "";
    } else {
      currentField += character;
    }
  }
  if (currentField !== "" || currentRow.length > 0) {
    currentRow.push(currentField);
    rows.push(currentRow);
  }
  return rows;
}

async function handleCsvFileSelected() {
  const selectedFile = csvFileInputElement.files[0];
  if (!selectedFile) return;

  const csvText = (await selectedFile.text()).replace(/^\uFEFF/, "");
  let rows = parseCsvIntoRows(csvText);
  if (csvHasHeaderCheckboxElement.checked) rows = rows.slice(1);

  const postsFromFile = rows.map((row) => (row[0] || "").trim()).filter((post) => post.length > 0);
  postInputElement.value = postsFromFile.join("\n---\n");
  document.querySelector('input[name="split-mode"][value="separator"]').checked = true;
  statusMessageElement.textContent = `Loaded ${postsFromFile.length} posts from ${selectedFile.name}.`;
}

function handleAnalyzeClicked() {
  const splitMode = document.querySelector('input[name="split-mode"]:checked').value;
  const listOfPosts = splitIntoPosts(postInputElement.value, splitMode);

  if (listOfPosts.length === 0) {
    statusMessageElement.textContent = "Please add at least one post.";
    return;
  }
  renderResults(analyzePostBatch(listOfPosts));
  statusMessageElement.textContent = `Analyzed ${listOfPosts.length} posts.`;
}


/* ---------- 5. Start-up ---------- */

async function loadModel() {
  try {
    const response = await fetch(MODEL_FILE_PATH);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const loadedModelBundle = await response.json();
    preparedTopicModel = prepareModel(loadedModelBundle.topic_model);
    preparedSentimentModel = prepareModel(loadedModelBundle.sentiment_model);

    analyzeButtonElement.disabled = false;
    statusMessageElement.textContent = "Model loaded. Add your posts and click Analyze.";
  } catch (error) {
    statusMessageElement.textContent =
      "Could not load solace_model.json. Serve this folder with a local web server " +
      "(for example: python -m http.server) instead of opening the file directly.";
    console.error(error);
  }
}

analyzeButtonElement.addEventListener("click", handleAnalyzeClicked);
csvFileInputElement.addEventListener("change", handleCsvFileSelected);
loadModel();
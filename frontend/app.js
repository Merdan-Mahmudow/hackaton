const API_BASE = '';
const form = document.getElementById('analyze-form');
const textarea = document.getElementById('feedback-text');
const resultBlock = document.getElementById('result');
const barsContainer = document.getElementById('probability-bars');
const historyList = document.getElementById('history');
const historyEmpty = document.getElementById('history-empty');
const totalCounter = document.getElementById('total-counter');
const modelName = document.getElementById('model-name');
const modelDetail = document.getElementById('model-detail');
const sessionDataset = document.getElementById('session-dataset');
const correctionForm = document.getElementById('correction-form');
const feedbackNotes = document.getElementById('feedback-notes');
const feedbackStatus = document.getElementById('feedback-status');
const bulkForm = document.getElementById('bulk-form');
const bulkFileInput = document.getElementById('bulk-file');
const bulkFileLabel = document.getElementById('bulk-file-label');
const bulkStatus = document.getElementById('bulk-status');
const bulkSummary = document.getElementById('bulk-summary');
const bulkTable = document.getElementById('bulk-table');
const bulkDownload = document.getElementById('bulk-download');
const metricAccuracy = document.getElementById('metric-accuracy');
const metricMacroF1 = document.getElementById('metric-macro-f1');
const metricsDatasetInfo = document.getElementById('metrics-dataset');
const metricsUpdated = document.getElementById('metrics-updated');
const metricsTableBody = document.getElementById('metrics-table-body');
const metricsConfusion = document.getElementById('metrics-confusion');
const labelBreakdown = document.getElementById('label-breakdown');
const macroF1Value = document.getElementById('macro-f1-value');
const macroF1Progress = document.getElementById('macro-f1-progress');
const macroF1Chip = document.getElementById('macro-f1-chip-value');
const macroF1Caption = document.getElementById('macro-f1-caption');
const historyTotal = document.getElementById('history-total');
const historyRange = document.getElementById('history-range');
const historyAverage = document.getElementById('history-average');
const historyLabelsList = document.getElementById('history-labels');
const historyDatesList = document.getElementById('history-dates');
const historyUpdated = document.getElementById('history-updated');
let chart;
let lastPrediction = null;
let lastAnalyzedText = '';
let bulkPredictions = [];

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Ошибка запроса: ${response.status}`);
  }
  return response.json();
}

function renderResult(prediction) {
  const { label, scores } = prediction;
  const mapping = {
    positive: { text: 'Положительная', class: 'result--positive' },
    negative: { text: 'Отрицательная', class: 'result--negative' },
    neutral: { text: 'Нейтральная', class: 'result--neutral' },
  };
  const { text, class: className } = mapping[label] || { text: label, class: 'result--neutral' };
  resultBlock.className = `result ${className}`;
  resultBlock.innerHTML = `<div><p class="muted">Модель определила тональность:</p><h3>${text}</h3></div>`;
  renderBars(scores);
  lastPrediction = prediction;
  setFeedbackEnabled(true);
  selectFeedbackLabel(label);
}

function setFeedbackStatus(message, state = 'info') {
  if (!feedbackStatus) return;
  const classes = ['feedback-form__status'];
  if (state === 'success') classes.push('feedback-form__status--success');
  if (state === 'error') classes.push('feedback-form__status--error');
  feedbackStatus.className = classes.join(' ');
  feedbackStatus.textContent = message;
}

function setFeedbackEnabled(enabled) {
  if (!correctionForm) return;
  correctionForm.classList.toggle('feedback-form--inactive', !enabled);
  const controls = correctionForm.querySelectorAll('input, textarea, button');
  controls.forEach((control) => {
    if (control === feedbackStatus) return;
    control.disabled = !enabled;
  });
  if (!enabled) {
    correctionForm.reset();
    lastPrediction = null;
    lastAnalyzedText = '';
  }
  setFeedbackStatus(
    enabled
      ? 'Если результат неверен, выберите правильную тональность и отправьте исправление.'
      : 'Сначала выполните анализ, чтобы отправить корректировку.'
  );
}

function selectFeedbackLabel(label) {
  if (!correctionForm) return;
  const radio = correctionForm.querySelector(`input[name="user-label"][value="${label}"]`);
  if (radio) {
    radio.checked = true;
  }
}

setFeedbackEnabled(false);

function renderBars(scores) {
  barsContainer.innerHTML = '';
  Object.entries(scores).forEach(([label, value]) => {
    const wrapper = document.createElement('div');
    const labelEl = document.createElement('div');
    labelEl.className = 'bar__label';
    labelEl.innerHTML = `<span>${label}</span><span>${(value * 100).toFixed(1)}%</span>`;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const span = document.createElement('span');
    span.style.width = `${(value * 100).toFixed(1)}%`;
    span.style.background = labelColor(label);
    bar.appendChild(span);
    wrapper.appendChild(labelEl);
    wrapper.appendChild(bar);
    barsContainer.appendChild(wrapper);
  });
}

function toggleHidden(element, visible) {
  if (!element) return;
  if (visible) {
    element.classList.remove('hidden');
  } else {
    element.classList.add('hidden');
  }
}

function labelColor(label) {
  switch (label) {
    case 'positive':
      return 'var(--positive)';
    case 'negative':
      return 'var(--negative)';
    default:
      return 'var(--neutral)';
  }
}

function formatPercent(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

function formatSupport(value) {
  if (typeof value === 'number') {
    return value.toString();
  }
  return '—';
}

function getMetricValue(entry, key) {
  if (!entry) return undefined;
  if (key in entry) return entry[key];
  const normalizedKey = key.replace('-', '_');
  return entry[normalizedKey];
}

function renderLabelDistribution(distribution) {
  if (!labelBreakdown) return;
  const entries = Object.entries(distribution || {});
  if (!entries.length) {
    labelBreakdown.innerHTML = '<p class="muted">Данные появятся после первых предсказаний.</p>';
    return;
  }
  labelBreakdown.innerHTML = entries
    .map(([label, value]) => {
      const percent = Math.round((value || 0) * 1000) / 10;
      return `
        <div class="label-breakdown__row">
          <div class="chip"><span class="dot" style="background:${labelColor(label)}"></span>${label}</div>
          <div class="progress-bar"><span style="width:${percent}%;background:${labelColor(label)}"></span></div>
          <strong>${percent}%</strong>
        </div>`;
    })
    .join('');
}

function updateHistory(history) {
  if (!historyList) return;
  historyList.innerHTML = '';
  if (!history?.length) {
    toggleHidden(historyEmpty, true);
    return;
  }
  toggleHidden(historyEmpty, false);
  history.forEach((item) => {
    const confidence = item.scores?.[item.label] ?? 0;
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${item.text}</td>
      <td><span class="chip" style="background:${labelColor(item.label)};color:#0f172a;border-color:${labelColor(item.label)}">${item.label}</span></td>
      <td>${(confidence * 100).toFixed(1)}%</td>
      <td>${new Date(item.timestamp).toLocaleTimeString()}</td>
    `;
    historyList.appendChild(row);
  });
}

function updateChart(distribution) {
  const labels = Object.keys(distribution);
  const data = Object.values(distribution).map((v) => Math.round(v * 100));
  const ctx = document.getElementById('stats-chart');
  if (!ctx) return;
  if (!chart) {
    chart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [
          {
            data,
            backgroundColor: labels.map((label) => labelColor(label)),
          },
        ],
      },
      options: {
        plugins: {
          legend: { position: 'bottom' },
        },
      },
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].backgroundColor = labels.map((label) => labelColor(label));
    chart.update();
  }
}

async function refreshStats() {
  try {
    const stats = await fetchJSON(`${API_BASE}/stats`);
    totalCounter.textContent = stats.total_predictions;
    updateHistory(stats.recent_predictions);
    updateChart(stats.label_distribution);
    renderLabelDistribution(stats.label_distribution);
  } catch (error) {
    console.error(error);
  }
}

async function loadModelInfo() {
  try {
    const info = await fetchJSON(`${API_BASE}/model`);
    modelName.textContent = info.algorithm || 'Модель';
    modelDetail.textContent = info.classes ? `Классы: ${info.classes.join(', ')}` : '';
  } catch (error) {
    modelName.textContent = 'Не удалось получить данные';
    console.error(error);
  }
}

function renderClassificationReport(report) {
  if (!metricsTableBody) return;
  const entries = Object.entries(report || {}).filter(([, value]) => typeof value === 'object');
  if (!entries.length) {
    metricsTableBody.innerHTML = `
      <tr>
        <td colspan="5" class="muted">Метрики появятся после запуска make evaluate.</td>
      </tr>
    `;
    return;
  }
  metricsTableBody.innerHTML = entries
    .map(([label, metrics]) => {
      const precision = formatPercent(getMetricValue(metrics, 'precision'));
      const recall = formatPercent(getMetricValue(metrics, 'recall'));
      const f1 = formatPercent(getMetricValue(metrics, 'f1-score') ?? getMetricValue(metrics, 'f1_score'));
      const support = formatSupport(metrics?.support);
      return `
        <tr>
          <td>${label}</td>
          <td>${precision}</td>
          <td>${recall}</td>
          <td>${f1}</td>
          <td>${support}</td>
        </tr>`;
    })
    .join('');
}

function renderConfusionMatrix(confusion) {
  if (!metricsConfusion) return;
  if (!confusion?.labels?.length || !confusion?.matrix?.length) {
    metricsConfusion.textContent = 'Матрица ошибок появится после запуска make evaluate.';
    return;
  }
  const rows = confusion.labels
    .map((label, rowIndex) => {
      const row = confusion.matrix[rowIndex] || [];
      const cells = row
        .map((value, colIndex) => `${confusion.labels[colIndex] || colIndex}: ${value}`)
        .join(', ');
      return `<li><strong>${label}</strong> → ${cells}</li>`;
    })
    .join('');
  metricsConfusion.innerHTML = `
    <strong>Матрица ошибок</strong>
    <ul class="history-list">${rows}</ul>
  `;
}

function renderCountsList(element, data, emptyMessage) {
  if (!element) return;
  const entries = Object.entries(data || {});
  if (!entries.length) {
    element.innerHTML = `<li class="muted">${emptyMessage}</li>`;
    return;
  }
  element.innerHTML = entries.map(([key, value]) => `<li><strong>${key}</strong>: ${value}</li>`).join('');
}

function formatDateRange(start, end) {
  if (!start && !end) return 'Нет отчёта';
  const toLocale = (value) =>
    value ? new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—';
  return `${toLocale(start)} — ${toLocale(end)}`;
}

function updateMacroF1(value) {
  const percent = Number.isFinite(value) ? Math.round(value * 100) : null;
  const display = Number.isFinite(percent) ? `${percent}%` : '—';
  if (macroF1Value) macroF1Value.textContent = display;
  if (macroF1Chip) macroF1Chip.textContent = display;
  if (macroF1Progress && Number.isFinite(percent)) {
    macroF1Progress.style.background = `conic-gradient(var(--accent) ${percent}%, rgba(255,255,255,0.08) ${percent}% 100%)`;
  }
}

async function loadEvalMetrics() {
  if (!metricAccuracy) return;
  try {
    const metrics = await fetchJSON(`${API_BASE}/reports/metrics`);
    metricAccuracy.textContent = formatPercent(metrics.accuracy);
    metricMacroF1.textContent = formatPercent(metrics.macro_f1);
    metricsDatasetInfo.textContent = metrics.dataset ? `Датасет: ${metrics.dataset}` : '';
    if (sessionDataset) sessionDataset.textContent = metrics.dataset || '—';
    metricsUpdated.textContent = metrics.generated_at
      ? `Обновлено: ${new Date(metrics.generated_at).toLocaleString()}`
      : '';
    if (macroF1Caption) {
      macroF1Caption.textContent = metrics.generated_at
        ? `Отчёт обновлён ${new Date(metrics.generated_at).toLocaleString()}`
        : 'Запустите make evaluate, чтобы обновить метрики.';
    }
    updateMacroF1(metrics.macro_f1);
    renderClassificationReport(metrics.classification_report);
    renderConfusionMatrix(metrics.confusion_matrix);
  } catch (error) {
    metricAccuracy.textContent = '—';
    metricMacroF1.textContent = '—';
    metricsDatasetInfo.textContent = 'Нет отчёта — запустите make evaluate';
    if (sessionDataset) sessionDataset.textContent = '—';
    metricsUpdated.textContent = error.message || '';
    if (macroF1Caption) macroF1Caption.textContent = 'Нет отчёта — запустите make evaluate.';
    updateMacroF1(null);
    if (metricsTableBody) {
      metricsTableBody.innerHTML = `
        <tr>
          <td colspan="5" class="muted">${error.message || 'Не удалось загрузить метрики.'}</td>
        </tr>
      `;
    }
    if (metricsConfusion) {
      metricsConfusion.textContent = '';
    }
  }
}

async function loadHistorySummary() {
  if (!historyTotal) return;
  try {
    const summary = await fetchJSON(`${API_BASE}/reports/history`);
    historyTotal.textContent = summary.total_predictions ?? 0;
    historyAverage.textContent = Math.round(summary.average_text_length ?? 0);
    historyRange.textContent = formatDateRange(summary.first_timestamp, summary.last_timestamp);
    historyUpdated.textContent = summary.generated_at
      ? `Обновлено: ${new Date(summary.generated_at).toLocaleString()}`
      : '';
    renderCountsList(historyLabelsList, summary.label_counts, 'Нет данных по классам.');
    renderCountsList(historyDatesList, summary.date_counts, 'Нет данных по датам.');
  } catch (error) {
    historyTotal.textContent = '0';
    historyAverage.textContent = '0';
    historyRange.textContent = 'Нет отчёта — выполните make history-report';
    historyUpdated.textContent = error.message || '';
    if (historyLabelsList) historyLabelsList.innerHTML = '';
    if (historyDatesList) historyDatesList.innerHTML = '';
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = textarea.value.trim();
  if (!text) return;
  form.classList.add('loading');
  lastAnalyzedText = text;
  try {
    const prediction = await fetchJSON(`${API_BASE}/predict`, {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
    renderResult(prediction);
    textarea.value = '';
    await refreshStats();
  } catch (error) {
    resultBlock.className = 'result result--negative';
    resultBlock.innerHTML = `<p>Ошибка: ${error.message}</p>`;
    setFeedbackEnabled(false);
  } finally {
    form.classList.remove('loading');
  }
});

if (correctionForm) {
  correctionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!lastPrediction || !lastAnalyzedText) {
      setFeedbackStatus('Сначала выполните анализ текста.', 'error');
      return;
    }
    const selected = correctionForm.querySelector('input[name="user-label"]:checked');
    if (!selected) {
      setFeedbackStatus('Выберите корректную тональность.', 'error');
      return;
    }
    const payload = {
      text: lastAnalyzedText,
      predicted_label: lastPrediction.label,
      user_label: selected.value,
      scores: lastPrediction.scores,
      notes: feedbackNotes && feedbackNotes.value.trim() ? feedbackNotes.value.trim() : undefined,
    };
    correctionForm.classList.add('loading');
    try {
      await fetchJSON(`${API_BASE}/feedback`, {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      setFeedbackStatus('Спасибо! Исправление сохранено.', 'success');
      correctionForm.reset();
      selectFeedbackLabel(lastPrediction.label);
    } catch (error) {
      setFeedbackStatus(`Ошибка: ${error.message}`, 'error');
    } finally {
      correctionForm.classList.remove('loading');
    }
  });
}

function setBulkStatus(message, state = 'info') {
  if (!bulkStatus) return;
  bulkStatus.textContent = message;
  const classes = ['bulk-form__status'];
  if (state === 'error') {
    classes.push('feedback-form__status--error');
  }
  bulkStatus.className = classes.join(' ');
}

function renderBulkSummary(summary) {
  if (!bulkSummary) return;
  const inputRows = summary?.input_rows ?? 0;
  const processedRows = summary?.processed_rows ?? 0;
  const skippedRows = summary?.skipped_rows ?? Math.max(0, inputRows - processedRows);
  const countsMarkup = Object.entries(summary?.class_counts || {})
    .map(([label, count]) => `<p><strong>${label}:</strong> ${count}</p>`)
    .join('');
  bulkSummary.innerHTML = `
    <div class="bulk-summary__card">
      <h4>Строк обработано</h4>
      <p><strong>${processedRows}</strong> из ${inputRows}</p>
      <p class="muted">Пропущено: ${skippedRows}</p>
    </div>
    <div class="bulk-summary__card">
      <h4>Распределение классов</h4>
      ${countsMarkup || '<p>Нет данных</p>'}
    </div>
  `;
  toggleHidden(bulkSummary, true);
}

function renderBulkTable(predictions) {
  if (!bulkTable) return;
  if (!predictions.length) {
    bulkTable.innerHTML = '';
    toggleHidden(bulkTable, false);
    return;
  }
  const rows = predictions
    .map(
      (pred) => `
      <tr>
        <td>${pred.row}</td>
        <td>${pred.label}</td>
        <td>${(pred.scores[pred.label] * 100).toFixed(1)}%</td>
        <td>${pred.text}</td>
      </tr>`
    )
    .join('');
  bulkTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Класс</th>
          <th>Уверенность</th>
          <th>Текст</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  toggleHidden(bulkTable, true);
}

function downloadBulkCSV() {
  if (!bulkPredictions.length) return;
  const header = ['row', 'label', 'confidence', 'text'];
  const lines = [header.join(',')];
  bulkPredictions.forEach((pred) => {
    const confidence = pred.scores[pred.label] ?? 0;
    const values = [pred.row, pred.label, confidence.toFixed(4), pred.text]
      .map((value) => {
        const stringValue = String(value ?? '');
        if (stringValue.includes('"') || stringValue.includes(',') || stringValue.includes('\n')) {
          return '"' + stringValue.replace(/"/g, '""') + '"';
        }
        return stringValue;
      })
      .join(',');
    lines.push(values);
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'predictions.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

if (bulkFileInput) {
  bulkFileInput.addEventListener('change', () => {
    if (!bulkFileLabel) return;
    const file = bulkFileInput.files && bulkFileInput.files[0];
    bulkFileLabel.textContent = file ? file.name : 'Выберите CSV-файл';
  });
}

if (bulkDownload) {
  bulkDownload.addEventListener('click', downloadBulkCSV);
}

if (bulkForm) {
  bulkForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = bulkFileInput && bulkFileInput.files ? bulkFileInput.files[0] : null;
    if (!file) {
      setBulkStatus('Выберите CSV-файл.', 'error');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    bulkForm.classList.add('loading');
    setBulkStatus('Файл отправлен, ждём результат…');
    try {
      const response = await fetch(`${API_BASE}/predict_file`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Не удалось обработать файл');
      }
      const payload = await response.json();
      bulkPredictions = payload.predictions || [];
      renderBulkSummary(payload.summary || {});
      renderBulkTable(bulkPredictions);
      toggleHidden(bulkDownload, bulkPredictions.length > 0);
      await refreshStats();
      setBulkStatus('Готово! Результаты можно скачать ниже.');
    } catch (error) {
      setBulkStatus(`Ошибка: ${error.message}`, 'error');
      toggleHidden(bulkSummary, false);
      toggleHidden(bulkTable, false);
      toggleHidden(bulkDownload, false);
    } finally {
      bulkForm.classList.remove('loading');
    }
  });
}

loadModelInfo();
refreshStats();
loadEvalMetrics();
loadHistorySummary();
setInterval(refreshStats, 5000);

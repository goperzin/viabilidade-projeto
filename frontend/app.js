const API_BASE = 'http://localhost:8000';

// =============================================================================
// UTILITÁRIOS DE UI
// =============================================================================

/**
 * Exibe o banner de erro global com a mensagem fornecida.
 * @param {string} msg
 */
function showError(msg) {
  const banner = document.getElementById('error-banner');
  const text   = document.getElementById('error-message');
  text.textContent = msg;
  banner.hidden = false;
}

/**
 * Oculta o banner de erro global.
 */
function hideError() {
  document.getElementById('error-banner').hidden = true;
}

/**
 * Coloca um botão no estado de carregamento.
 * @param {HTMLButtonElement} btn
 * @param {string} label - Texto exibido durante o carregamento.
 */
function showLoading(btn, label = 'Aguarde...') {
  btn.disabled = true;
  btn.classList.add('btn--loading');
  btn.dataset.originalText = btn.textContent;
  btn.textContent = label;
}

/**
 * Restaura o botão ao estado normal.
 * @param {HTMLButtonElement} btn
 */
function resetLoading(btn) {
  btn.disabled = false;
  btn.classList.remove('btn--loading');
  btn.textContent = btn.dataset.originalText || btn.textContent;
}

// =============================================================================
// CHAMADAS DE API
// =============================================================================

/**
 * Busca o status do modelo via GET /status.
 * Mostra/oculta seções conforme model_trained.
 */
async function checkStatus() {
  hideError();
  try {
    const res  = await fetch(`${API_BASE}/status`);
    const data = await res.json();

    const badge = document.getElementById('status-badge');

    if (data.model_trained) {
      badge.textContent = 'Modelo treinado';
      badge.className   = 'badge badge--trained';

      document.getElementById('upload-section').hidden  = true;
      document.getElementById('metrics-section').hidden = false;
      document.getElementById('predict-section').hidden = false;

      renderMetrics(data.metrics);
    } else {
      badge.textContent = 'Modelo não treinado';
      badge.className   = 'badge badge--untrained';

      document.getElementById('upload-section').hidden  = false;
      document.getElementById('metrics-section').hidden = true;
      document.getElementById('predict-section').hidden = true;
      document.getElementById('result-section').hidden  = true;
    }
  } catch (_err) {
    showError('Não foi possível conectar ao servidor. Verifique se o backend está rodando.');
    document.getElementById('status-badge').textContent = 'Sem conexão';
    document.getElementById('status-badge').className   = 'badge badge--untrained';
  }
}

/**
 * Envia o CSV via POST /train e exibe feedback.
 * @param {Event} e
 */
async function handleUpload(e) {
  e.preventDefault();
  hideError();

  const fileInput    = document.getElementById('csv-file');
  const btn          = document.getElementById('upload-btn');
  const feedbackEl   = document.getElementById('upload-feedback');

  feedbackEl.hidden = true;
  feedbackEl.className = 'feedback';

  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  showLoading(btn, 'Treinando...');

  try {
    const res  = await fetch(`${API_BASE}/train`, { method: 'POST', body: formData });
    const data = await res.json();

    if (res.ok && data.success) {
      feedbackEl.textContent = data.message || 'Modelo treinado com sucesso.';
      feedbackEl.classList.add('feedback--success');
      feedbackEl.hidden = false;
      await checkStatus();
    } else {
      const msg = data.detail || data.message || 'Erro ao treinar o modelo.';
      feedbackEl.textContent = msg;
      feedbackEl.classList.add('feedback--error');
      feedbackEl.hidden = false;
    }
  } catch (_err) {
    showError('Erro de conexão ao enviar o arquivo.');
  } finally {
    resetLoading(btn);
  }
}

/**
 * Lê o formulário e envia via POST /predict; chama renderResult com o retorno.
 * @param {Event} e
 */
async function handlePredict(e) {
  e.preventDefault();
  hideError();

  const btn = document.getElementById('predict-btn');

  const investment     = parseFloat(document.getElementById('investment').value);
  const expectedReturn = parseFloat(document.getElementById('expected_return').value);
  const impactScore    = parseFloat(document.getElementById('impact_score').value);

  showLoading(btn, 'Analisando...');

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        investment,
        expected_return: expectedReturn,
        impact_score:    impactScore,
      }),
    });

    const data = await res.json();

    if (res.ok) {
      renderResult(data);
    } else {
      showError(data.detail || 'Erro ao obter previsão.');
    }
  } catch (_err) {
    showError('Erro de conexão ao enviar os dados do projeto.');
  } finally {
    resetLoading(btn);
  }
}

// =============================================================================
// RENDERIZAÇÃO
// =============================================================================

/**
 * Preenche o painel de métricas com os dados retornados pela API.
 * Formata valores decimais como percentuais (0.89 → "89%").
 * @param {Object} metrics
 */
function renderMetrics(metrics) {
  const grid = document.getElementById('metrics-grid');
  if (!metrics || !grid) return;

  const fmt = (v) =>
    typeof v === 'number' && v <= 1 ? `${Math.round(v * 100)}%` : String(v);

  const items = [
    { label: 'Acurácia',              value: metrics.accuracy },
    { label: 'Precisão (classe 1)',   value: metrics['1']?.precision },
    { label: 'Recall (classe 1)',     value: metrics['1']?.recall },
    { label: 'F1-score (classe 1)',   value: metrics['1']?.['f1-score'] },
  ].filter((item) => item.value !== undefined);

  grid.innerHTML = items
    .map(
      (item) => `
      <div class="metric-item">
        <span class="metric-item__label">${item.label}</span>
        <span class="metric-item__value">${fmt(item.value)}</span>
      </div>`
    )
    .join('');
}

/**
 * Exibe o card de resultado com viabilidade, label e barra de probabilidade.
 * @param {{ viability: number, viability_label: string, probability: number }} data
 */
function renderResult(data) {
  const section      = document.getElementById('result-section');
  const badge        = document.getElementById('result-badge');
  const fill         = document.getElementById('progress-fill');
  const probValue    = document.getElementById('probability-value');

  const isViable = data.viability === 1;
  const pct      = Math.round(data.probability * 100);

  badge.textContent = data.viability_label;
  badge.className   = isViable ? 'badge badge--viable' : 'badge badge--inviable';

  // Atualizar barra de progresso
  fill.style.width = `${pct}%`;
  fill.style.backgroundColor = isViable
    ? 'var(--color-viable)'
    : 'var(--color-inviable)';

  probValue.textContent = `${pct}%`;

  // Atualizar aria para acessibilidade
  const progressBar = fill.closest('.progress-bar');
  if (progressBar) {
    progressBar.setAttribute('aria-valuenow', pct);
  }

  section.hidden = false;
  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// =============================================================================
// EVENT LISTENERS E INICIALIZAÇÃO
// =============================================================================

document.getElementById('upload-form').addEventListener('submit', handleUpload);
document.getElementById('predict-form').addEventListener('submit', handlePredict);

// Exibe o nome do arquivo selecionado
document.getElementById('csv-file').addEventListener('change', (e) => {
  const fileNameEl = document.getElementById('file-name');
  if (e.target.files.length) {
    fileNameEl.textContent = e.target.files[0].name;
    fileNameEl.hidden = false;
  } else {
    fileNameEl.hidden = true;
  }
});

// Verifica o status ao carregar a página
checkStatus();

(() => {
  'use strict';

  const metrics = {
    f1_macro: ['F1 macro', 'Promedio del F1 de cada clase. Cada clase tiene el mismo peso.'],
    accuracy: ['Accuracy', 'Proporción de registros o ventanas clasificados correctamente.'],
    precision_macro: ['Precisión macro', 'Promedio por clase de la proporción de predicciones positivas correctas.'],
    recall_macro: ['Recall macro', 'Promedio por clase de la proporción de casos reales que se detectaron.'],
    auc: ['AUC ROC', 'Capacidad de separar clases al variar el umbral. En etapa 2 se usa el promedio macro uno contra el resto.'],
  };
  const tableColumns = ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'auc'];
  const percent = new Intl.NumberFormat('es-PY', { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const all = window.pulsoMetricas;
  if (!all || ![1, 2].every(stage => all[stage]?.length === 6)) return;

  const panel = document.querySelector('#explorador');
  const stageButtons = [...panel.querySelectorAll('[data-stage]')];
  const picker = document.querySelector('#metrica');
  const bars = document.querySelector('#barras');
  const rows = document.querySelector('#tabla-metricas');
  const summary = document.querySelector('#grafico-resumen');
  const explanation = document.querySelector('#metrica-explicacion');
  const caption = document.querySelector('#caption-tabla');
  let stage = '1';

  function cell(kind, value, className) {
    const element = document.createElement(kind);
    element.textContent = value;
    if (className) element.className = className;
    return element;
  }

  function render() {
    const key = picker.value;
    const [name, definition] = metrics[key];
    const ranked = [...all[stage]].sort((a, b) => b[key] - a[key]);
    const best = ranked[0][key];
    const leaders = ranked.filter(model => model[key] === best);
    panel.dataset.stage = stage;
    stageButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.stage === stage)));
    summary.textContent = 'Etapa ' + stage + ' · ' + name + ' · Mejor valor: ' + leaders.map(model => model.name).join(' y ') + ' (' + percent.format(best) + ').';
    explanation.textContent = definition;
    caption.textContent = 'Etapa ' + stage + ' · Métricas por modelo, ordenadas por ' + name;
    bars.replaceChildren();
    rows.replaceChildren();
    for (const model of ranked) {
      const row = cell('li', '', model[key] === best ? 'bar-row is-best' : 'bar-row');
      const track = cell('div', '', 'bar-track');
      track.setAttribute('aria-hidden', 'true');
      const fill = cell('div', '', 'bar-fill');
      fill.style.width = (model[key] * 100) + '%';
      track.append(fill);
      row.append(cell('span', model.name, 'bar-label'), track, cell('span', percent.format(model[key]), 'bar-value'));
      bars.append(row);

      const tableRow = document.createElement('tr');
      const heading = cell('th', model.name);
      heading.scope = 'row';
      tableRow.append(heading, ...tableColumns.map(column => cell('td', percent.format(model[column]))));
      rows.append(tableRow);
    }
  }

  stageButtons.forEach(button => button.addEventListener('click', () => {
    stage = button.dataset.stage;
    render();
  }));
  picker.addEventListener('change', render);
  render();
  panel.hidden = false;
  document.querySelector('#grafico-estatico').hidden = true;
})();

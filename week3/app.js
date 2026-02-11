/**
 * Neural Network Design: The Gradient Puzzle
 * ------------------------------------------------------------
 * Mission: Transform random noise into a smooth, directional gradient
 * WITHOUT using target labels. Only rearrange existing pixels.
 *
 * Constraint: Input histogram ≈ Output histogram (no new colors).
 * Analogy: Sliding puzzle – move tiles, don't paint over them.
 *
 * ============= СТУДЕНЧЕСКИЕ ЗАДАНИЯ =============
 * [TODO-A] Архитектура: реализовать 'transformation' и 'expansion' в createStudentModel()
 * [TODO-B] Функция потерь: добавить smoothness + direction в studentLoss()
 * [TODO-C] Сравнить baseline и student (визуально + loss)
 * ================================================
 */

// ------------------------------------------------------------
// 1. ГЛОБАЛЬНОЕ СОСТОЯНИЕ И КОНФИГУРАЦИЯ
// ------------------------------------------------------------
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.02,
  autoTrainSpeed: 100,
  smoothnessWeight: 0.2,
  directionWeight: 0.3,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  currentArch: 'compression',
};

// ------------------------------------------------------------
// 2. ФУНКЦИИ ПОТЕРЬ (реализованы полностью)
// ------------------------------------------------------------

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

function smoothness(yPred) {
  return tf.tidy(() => {
    const batchSize = yPred.shape[0];
    const height = yPred.shape[1];
    const width = yPred.shape[2];
    const channels = yPred.shape[3];
    
    // Разница по горизонтали
    const left = yPred.slice([0, 0, 0, 0], [batchSize, height, width - 1, channels]);
    const right = yPred.slice([0, 0, 1, 0], [batchSize, height, width - 1, channels]);
    const diffX = left.sub(right);

    // Разница по вертикали
    const top = yPred.slice([0, 0, 0, 0], [batchSize, height - 1, width, channels]);
    const bottom = yPred.slice([0, 1, 0, 0], [batchSize, height - 1, width, channels]);
    const diffY = top.sub(bottom);

    return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
  });
}

function directionX(yPred) {
  return tf.tidy(() => {
    const width = 16;
    // Маска: слева -1, справа +1
    const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
    // Отрицательная корреляция - чем ярче справа, тем меньше loss
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// ------------------------------------------------------------
// 3. АРХИТЕКТУРА МОДЕЛЕЙ
// ------------------------------------------------------------

// Baseline модель - всегда компрессия, всегда MSE
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// -----------------------------------------------------------------
// [TODO-A] СТУДЕНЧЕСКАЯ АРХИТЕКТУРА
// Раскомментируй нужные строки для каждого типа архитектуры
// -----------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === 'compression') {
    // Компрессия: 256 -> 64 -> 256
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));

  } else if (archType === 'transformation') {
    // ========== [TODO-A] ТРАНСФОРМАЦИЯ ==========
    // 1:1 отображение, размерность сохраняется
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));

  } else if (archType === 'expansion') {
    // ========== [TODO-A] ЭКСПАНСИЯ ==========
    // Расширение: 256 -> 512 -> 256
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));

  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------
// 4. КАСТОМНАЯ ФУНКЦИЯ ПОТЕРЬ - [TODO-B]
// ------------------------------------------------------------
// -----------------------------------------------------------------
// [TODO-B] СТУДЕНЧЕСКАЯ ФУНКЦИЯ ПОТЕРЬ
// Раскомментируй строки с smoothness и direction, закомментируй return lossMSE
// -----------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. Реконструкция - сохраняем пиксели
    const lossMSE = mse(yTrue, yPred);

    // ========== [TODO-B] АКТИВИРУЙ КАСТОМНЫЕ ПОТЕРИ ==========
    // 2. Сглаженность - соседние пиксели похожи
    const lossSmooth = smoothness(yPred).mul(CONFIG.smoothnessWeight);
    
    // 3. Направление - слева темно, справа светло
    const lossDir = directionX(yPred).mul(CONFIG.directionWeight);
    
    // TOTAL LOSS: ВСЕ ТРИ КОМПОНЕНТЫ
    return lossMSE.add(lossSmooth).add(lossDir);
    
    // ---------- DEFAULT: ТОЛЬКО MSE ----------
    // return lossMSE;
  });
}

// ------------------------------------------------------------
// 5. ТРЕНИРОВКА
// ------------------------------------------------------------
async function trainStep() {
  if (!state.studentModel || !state.baselineModel) {
    resetModels();
    return;
  }

  state.step++;

  // Тренируем baseline
  const baselineLoss = tf.tidy(() => {
    const optimizer = tf.train.adam(CONFIG.learningRate);
    const yPred = state.baselineModel.predict(state.xInput);
    const loss = mse(state.xInput, yPred);
    
    const grads = tf.grads(() => {
      const pred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, pred);
    });
    
    const gradients = grads([], state.baselineModel.getWeights());
    optimizer.applyGradients(gradients.map((g, i) => ({ 
      name: state.baselineModel.getWeights()[i].name, 
      tensor: g 
    })), state.baselineModel.getWeights());
    
    return loss.dataSync()[0];
  });

  // Тренируем student
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const optimizer = tf.train.adam(CONFIG.learningRate);
      const yPred = state.studentModel.predict(state.xInput);
      const loss = studentLoss(state.xInput, yPred);
      
      const grads = tf.grads(() => {
        const pred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, pred);
      });
      
      const gradients = grads([], state.studentModel.getWeights());
      optimizer.applyGradients(gradients.map((g, i) => ({ 
        name: state.studentModel.getWeights()[i].name, 
        tensor: g 
      })), state.studentModel.getWeights());
      
      return loss.dataSync()[0];
    });

    log(`✅ Шаг ${state.step} | Baseline: ${baselineLoss.toFixed(5)} | Student: ${studentLossVal.toFixed(5)}`);
  } catch (e) {
    log(`❌ Ошибка: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Рендерим каждый шаг
  await render();
  updateLossDisplay(baselineLoss, studentLossVal);
}

// ------------------------------------------------------------
// 6. РЕНДЕРИНГ
// ------------------------------------------------------------
async function render() {
  if (!state.baselineModel || !state.studentModel) return;
  
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(basePred.squeeze(), document.getElementById('canvas-baseline'));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById('canvas-student'));

  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(baseLoss, studLoss) {
  document.getElementById('loss-baseline').innerText = `Loss: ${baseLoss.toFixed(5)}`;
  document.getElementById('loss-student').innerText = `Loss: ${studLoss.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById('log-area');
  const entry = document.createElement('div');
  entry.innerText = `> ${msg}`;
  if (isError) entry.classList.add('error');
  el.prepend(entry);
  if (el.children.length > 8) el.removeChild(el.lastChild);
}

// ------------------------------------------------------------
// 7. СБРОС И ИНИЦИАЛИЗАЦИЯ
// ------------------------------------------------------------
function resetModels(archType = null) {
  if (typeof archType !== 'string') {
    archType = document.querySelector('input[name="arch"]:checked')?.value || 'compression';
  }
  
  state.currentArch = archType;
  
  if (state.isAutoTraining) stopAutoTrain();
  
  // Очищаем память
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  
  // Создаем новые модели
  state.baselineModel = createBaselineModel();
  
  try {
    state.studentModel = createStudentModel(archType);
    // Прогоняем dummy data для построения графа
    state.studentModel.predict(tf.zeros([1, 16, 16, 1]));
    log(`🔄 Архитектура студента: ${archType}`);
  } catch (e) {
    log(`⚠️ Ошибка создания ${archType}: ${e.message}`, true);
    state.studentModel = createStudentModel('compression');
    state.studentModel.predict(tf.zeros([1, 16, 16, 1]));
    log('⚡ Используем Compression', true);
  }
  
  state.step = 0;
  
  document.getElementById('student-arch-label').innerText = 
    archType.charAt(0).toUpperCase() + archType.slice(1);
  
  render();
  updateLossDisplay(0, 0);
}

function init() {
  tf.setBackend('cpu');
  
  // Фиксированный шум
  state.xInput = tf.randomUniform(CONFIG.inputShapeData, 0, 1, 'float32');
  
  // Рендерим вход
  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById('canvas-input'));

  // События
  document.getElementById('btn-train').addEventListener('click', () => trainStep());
  document.getElementById('btn-auto').addEventListener('click', toggleAutoTrain);
  document.getElementById('btn-reset').addEventListener('click', () => resetModels());

  document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      resetModels(e.target.value);
    });
  });

  resetModels('compression');
  
  log('🚀 Готово! Выбери архитектуру и начни тренировку');
  log('💡 Совет: Выбери "Transformation" и тренируй 50+ шагов');
  log('🎯 Цель: Сглаженный градиент слева направо');
}

// ------------------------------------------------------------
// 8. АВТО-ТРЕНИРОВКА
// ------------------------------------------------------------
function toggleAutoTrain() {
  const btn = document.getElementById('btn-auto');
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = '⏸️ Стоп';
    btn.classList.add('btn-stop');
    autoTrainLoop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById('btn-auto');
  btn.innerText = '▶️ Авто';
  btn.classList.remove('btn-stop');
}

function autoTrainLoop() {
  if (state.isAutoTraining) {
    trainStep().then(() => {
      setTimeout(autoTrainLoop, CONFIG.autoTrainSpeed);
    });
  }
}

// Запуск
init();
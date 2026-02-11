/**
 * Neural Network Design: The Gradient Puzzle
 * ------------------------------------------------------------
 * Mission: Transform random noise into a smooth, directional gradient
 * WITHOUT using target labels. Only rearrange existing pixels.
 */

// ------------------------------------------------------------
// 1. ГЛОБАЛЬНОЕ СОСТОЯНИЕ
// ------------------------------------------------------------
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.01,
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
  baselineWeights: null,
  studentWeights: null,
  currentArch: 'compression',
};

// ------------------------------------------------------------
// 2. ФУНКЦИИ ПОТЕРЬ
// ------------------------------------------------------------
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

function smoothness(yPred) {
  return tf.tidy(() => {
    const diffX = yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
    const diffY = yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
    return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
  });
}

function directionX(yPred) {
  return tf.tidy(() => {
    const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// ------------------------------------------------------------
// 3. МОДЕЛИ
// ------------------------------------------------------------
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));

  if (archType === 'compression') {
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } 
  else if (archType === 'transformation') {
    // [TODO-A] Трансформация - сохраняем размерность
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } 
  else if (archType === 'expansion') {
    // [TODO-A] Экспансия - увеличиваем размерность
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------
// 4. LOSS ФУНКЦИЯ ДЛЯ СТУДЕНТА [TODO-B]
// ------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossMSE = mse(yTrue, yPred);
    const lossSmooth = smoothness(yPred).mul(CONFIG.smoothnessWeight);
    const lossDir = directionX(yPred).mul(CONFIG.directionWeight);
    
    // [TODO-B] Раскомментируй для кастомных потерь:
    return lossMSE.add(lossSmooth).add(lossDir);
    
    // [TODO-B] Закомментируй это когда включишь кастомные потери:
    // return lossMSE;
  });
}

// ------------------------------------------------------------
// 5. ТРЕНИРОВКА - ПРОСТАЯ И РАБОЧАЯ
// ------------------------------------------------------------
async function trainStep() {
  if (!state.studentModel || !state.baselineModel) {
    resetModels();
    return;
  }

  state.step++;

  // Сохраняем веса если их нет
  if (!state.baselineWeights) {
    state.baselineWeights = state.baselineModel.getWeights().map(w => w.clone());
  }
  if (!state.studentWeights) {
    state.studentWeights = state.studentModel.getWeights().map(w => w.clone());
  }

  // Тренируем baseline
  tf.tidy(() => {
    const optimizer = tf.train.adam(CONFIG.learningRate);
    
    const loss = () => {
      const pred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, pred);
    };
    
    const grads = tf.grads(loss);
    const gradients = grads([], state.baselineWeights);
    
    optimizer.applyGradients(gradients.map((g, i) => ({ 
      name: i.toString(), 
      tensor: g 
    })), state.baselineWeights);
    
    // Обновляем веса модели
    state.baselineModel.setWeights(state.baselineWeights);
  });

  // Тренируем студента
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const optimizer = tf.train.adam(CONFIG.learningRate);
      
      const loss = () => {
        const pred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, pred);
      };
      
      const grads = tf.grads(loss);
      const gradients = grads([], state.studentWeights);
      
      optimizer.applyGradients(gradients.map((g, i) => ({ 
        name: i.toString(), 
        tensor: g 
      })), state.studentWeights);
      
      // Обновляем веса модели
      state.studentModel.setWeights(state.studentWeights);
      
      const pred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, pred).dataSync()[0];
    });

    log(`✅ Шаг ${state.step} | Студент loss: ${studentLossVal.toFixed(5)}`);
  } catch (e) {
    log(`❌ ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Рендер
  await render();
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

  const baseLoss = mse(state.xInput, basePred).dataSync()[0];
  const studLoss = studentLoss(state.xInput, studPred).dataSync()[0];
  
  document.getElementById('loss-baseline').innerText = `Loss: ${baseLoss.toFixed(5)}`;
  document.getElementById('loss-student').innerText = `Loss: ${studLoss.toFixed(5)}`;

  basePred.dispose();
  studPred.dispose();
}

// ------------------------------------------------------------
// 7. ЛОГИРОВАНИЕ
// ------------------------------------------------------------
function log(msg, isError = false) {
  const el = document.getElementById('log-area');
  const entry = document.createElement('div');
  entry.innerText = `> ${msg}`;
  if (isError) entry.classList.add('error');
  el.prepend(entry);
  if (el.children.length > 6) el.removeChild(el.lastChild);
}

// ------------------------------------------------------------
// 8. СБРОС МОДЕЛЕЙ
// ------------------------------------------------------------
function resetModels(archType = null) {
  if (typeof archType !== 'string') {
    archType = document.querySelector('input[name="arch"]:checked')?.value || 'compression';
  }
  
  state.currentArch = archType;
  
  if (state.isAutoTraining) stopAutoTrain();
  
  // Очищаем всё
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.baselineWeights) state.baselineWeights.forEach(w => w.dispose());
  if (state.studentWeights) state.studentWeights.forEach(w => w.dispose());
  
  // Создаём новые модели
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  
  // Инициализируем веса
  state.baselineWeights = state.baselineModel.getWeights().map(w => w.clone());
  state.studentWeights = state.studentModel.getWeights().map(w => w.clone());
  
  state.step = 0;
  
  document.getElementById('student-arch-label').innerText = 
    archType.charAt(0).toUpperCase() + archType.slice(1);
  
  log(`🔄 Архитектура: ${archType}`);
  render();
}

// ------------------------------------------------------------
// 9. ИНИЦИАЛИЗАЦИЯ
// ------------------------------------------------------------
function init() {
  tf.setBackend('cpu');
  
  // Фиксированный шум
  state.xInput = tf.randomUniform([1, 16, 16, 1], 0, 1, 'float32');
  
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
  
  log('🚀 Погнали! Выбери архитектуру:');
  log('1. Compression - как baseline');
  log('2. Transformation - сохраняет размер (256)');
  log('3. Expansion - расширяет (512)');
  log('💡 В studentLoss() включи smoothness и direction!');
}

// ------------------------------------------------------------
// 10. АВТОТРЕНИРОВКА
// ------------------------------------------------------------
function toggleAutoTrain() {
  const btn = document.getElementById('btn-auto');
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = '⏸️ Стоп';
    btn.classList.add('btn-stop');
    trainLoop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById('btn-auto');
  btn.innerText = '▶️ Авто';
  btn.classList.remove('btn-stop');
}

function trainLoop() {
  if (state.isAutoTraining) {
    trainStep().then(() => {
      setTimeout(trainLoop, CONFIG.autoTrainSpeed);
    });
  }
}

// Старт
init();

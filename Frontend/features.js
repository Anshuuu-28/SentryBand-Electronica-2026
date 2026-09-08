/**
 * SentryBand -- client-side feature extraction + model inference.
 *
 * A JavaScript port of src/features.py and the trained model's forward
 * pass, so a judge can upload their own sensor CSV and get a REAL
 * prediction from SentryBand's actual trained weights, entirely in the
 * browser -- no server, matching the whole product's "no cloud" story.
 *
 * HONESTY NOTE (documented, same spirit as the rest of this project):
 * the heart-rate peak-detection function below (findPeaksApprox) is a
 * close approximation of scipy.signal.find_peaks (distance + prominence
 * filtering), not a byte-for-byte reimplementation. For clearly-defined
 * peaks (real heartbeats) it agrees closely; on ambiguous/noisy signals
 * it may pick a different peak count than the Python version. Every
 * other feature (FFT-based dominant frequency, spectral energy ratio,
 * all time-domain stats) is computed with the exact same math as
 * src/features.py, verified numerically against it (see
 * scripts/verify_features_js.py).
 */

const SAMPLE_RATE_HZ = 25;
const WINDOW_SECONDS = 2.0;
const WINDOW_SAMPLES = Math.round(SAMPLE_RATE_HZ * WINDOW_SECONDS);

// ---------------------------------------------------------------------
// Resampling (exact port of real_data_loader.py's _resample -- linear
// interpolation, identical math)
// ---------------------------------------------------------------------
function linspace(start, stop, num) {
  const step = (stop - start) / num;
  const out = new Array(num);
  for (let i = 0; i < num; i++) out[i] = start + step * i;
  return out;
}

function interp1d(xNew, xOld, yOld) {
  const n = xOld.length;
  const out = new Array(xNew.length);
  for (let i = 0; i < xNew.length; i++) {
    const x = xNew[i];
    if (x <= xOld[0]) { out[i] = yOld[0]; continue; }
    if (x >= xOld[n - 1]) { out[i] = yOld[n - 1]; continue; }
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xOld[mid] <= x) lo = mid; else hi = mid;
    }
    const t = (x - xOld[lo]) / (xOld[hi] - xOld[lo]);
    out[i] = yOld[lo] + t * (yOld[hi] - yOld[lo]);
  }
  return out;
}

/** Resamples a 1-D array from srcHz to dstHz (linear interpolation). */
function resample1D(x, srcHz, dstHz) {
  const nSrc = x.length;
  const duration = nSrc / srcHz;
  const nDst = Math.round(duration * dstHz);
  const tSrc = linspace(0, duration, nSrc);
  const tDst = linspace(0, duration, nDst);
  return interp1d(tDst, tSrc, x);
}

/** Resamples an [n][3] array (accelerometer) from srcHz to dstHz. */
function resample3D(x, srcHz, dstHz) {
  const cols = [0, 1, 2].map((c) => x.map((row) => row[c]));
  const resampledCols = cols.map((col) => resample1D(col, srcHz, dstHz));
  const n = resampledCols[0].length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = [resampledCols[0][i], resampledCols[1][i], resampledCols[2][i]];
  return out;
}

// ---------------------------------------------------------------------
// Direct DFT (not FFT) -- window sizes here are tiny (~50-320 samples),
// so an O(n^2) direct DFT is instant in a browser and, unlike a radix-2
// FFT, needs no power-of-2 zero-padding -- exactly matching numpy's
// rfft on an arbitrary-length real signal, with no approximation.
// ---------------------------------------------------------------------
function rfftMagnitude(signal) {
  const n = signal.length;
  const nFreqs = Math.floor(n / 2) + 1;
  const mags = new Array(nFreqs);
  for (let k = 0; k < nFreqs; k++) {
    let re = 0, im = 0;
    const w = (2 * Math.PI * k) / n;
    for (let t = 0; t < n; t++) {
      re += signal[t] * Math.cos(w * t);
      im -= signal[t] * Math.sin(w * t);
    }
    mags[k] = Math.hypot(re, im);
  }
  return mags;
}

function rfftPower(signal) {
  const mags = rfftMagnitude(signal);
  return mags.map((m) => m * m);
}

function rfftFreqs(n, fs) {
  const nFreqs = Math.floor(n / 2) + 1;
  const out = new Array(nFreqs);
  for (let k = 0; k < nFreqs; k++) out[k] = (k * fs) / n;
  return out;
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function dominantFrequency(signal, fs) {
  const m = mean(signal);
  const centered = signal.map((v) => v - m);
  if (centered.length < 4 || centered.every((v) => Math.abs(v) < 1e-12)) return 0.0;
  const freqs = rfftFreqs(centered.length, fs);
  const spectrum = rfftMagnitude(centered);
  if (spectrum.length <= 1) return 0.0;
  let bestIdx = 1, bestVal = spectrum[1];
  for (let i = 2; i < spectrum.length; i++) {
    if (spectrum[i] > bestVal) { bestVal = spectrum[i]; bestIdx = i; }
  }
  return freqs[bestIdx];
}

function spectralEnergyRatio(signal, fs, bandLow = 0.5, bandHigh = 3.5) {
  const m = mean(signal);
  const centered = signal.map((v) => v - m);
  if (centered.length < 4) return 0.0;
  const freqs = rfftFreqs(centered.length, fs);
  const power = rfftPower(centered);
  let total = 1e-9;
  for (let i = 1; i < power.length; i++) total += power[i];
  let bandEnergy = 0;
  for (let i = 0; i < power.length; i++) {
    if (freqs[i] >= bandLow && freqs[i] <= bandHigh) bandEnergy += power[i];
  }
  return bandEnergy / total;
}

// ---------------------------------------------------------------------
// Accelerometer features -- exact port of extract_accel_features()
// ---------------------------------------------------------------------
function extractAccelFeatures(window, fs) {
  const mag = window.map(([x, y, z]) => Math.sqrt(x * x + y * y + z * z));
  const magMean = mean(mag);
  const magStd = Math.sqrt(mean(mag.map((v) => (v - magMean) ** 2)));
  const magMin = Math.min(...mag);
  const magMax = Math.max(...mag);
  const magPtp = magMax - magMin;

  let jerkMax = 0.0;
  if (mag.length > 1) {
    for (let i = 1; i < mag.length; i++) {
      const jerk = Math.abs((mag[i] - mag[i - 1]) * fs);
      if (jerk > jerkMax) jerkMax = jerk;
    }
  }

  const sma = mean(window.map(([x, y, z]) => Math.abs(x) + Math.abs(y) + Math.abs(z)));
  const dominantFreq = dominantFrequency(mag, fs);

  let zeroCrossings = 0;
  if (mag.length > 1) {
    const centered = mag.map((v) => v - magMean);
    for (let i = 1; i < centered.length; i++) {
      const s0 = Math.sign(centered[i - 1]);
      const s1 = Math.sign(centered[i]);
      if (s0 !== s1) zeroCrossings++;
    }
  }
  const zeroCrossRate = mag.length > 1 ? zeroCrossings / (mag.length - 1) : 0.0;

  return [magMean, magStd, magMin, magMax, magPtp, jerkMax, sma, dominantFreq, zeroCrossRate];
}

// ---------------------------------------------------------------------
// Peak detection -- approximation of scipy.signal.find_peaks
// (distance + prominence filtering). See HONESTY NOTE at top of file.
// ---------------------------------------------------------------------
function findPeaksApprox(signal, minDistance, minProminence) {
  const n = signal.length;
  const candidates = [];
  for (let i = 1; i < n - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]) candidates.push(i);
  }

  const withProminence = candidates.map((i) => {
    let leftMin = signal[i];
    for (let j = i - 1; j >= 0 && signal[j] <= signal[i]; j--) leftMin = Math.min(leftMin, signal[j]);
    let rightMin = signal[i];
    for (let j = i + 1; j < n && signal[j] <= signal[i]; j++) rightMin = Math.min(rightMin, signal[j]);
    const base = Math.max(leftMin, rightMin);
    return { i, prominence: signal[i] - base };
  }).filter((p) => p.prominence >= minProminence);

  withProminence.sort((a, b) => signal[b.i] - signal[a.i]);
  const accepted = [];
  for (const p of withProminence) {
    if (accepted.every((a) => Math.abs(a - p.i) >= minDistance)) accepted.push(p.i);
  }
  accepted.sort((a, b) => a - b);
  return accepted;
}

// ---------------------------------------------------------------------
// PPG features -- exact port of extract_ppg_features(), except peak
// detection (see above)
// ---------------------------------------------------------------------
function extractPpgFeatures(window, fs) {
  const minDistance = Math.max(1, Math.floor((fs * 60.0) / 220.0));
  const peaks = findPeaksApprox(window, minDistance, 0.3);

  let estHrBpm, rrStd;
  if (peaks.length >= 2) {
    const rrIntervals = [];
    for (let i = 1; i < peaks.length; i++) rrIntervals.push((peaks[i] - peaks[i - 1]) / fs);
    const rrMean = mean(rrIntervals);
    estHrBpm = 60.0 / rrMean;
    rrStd = Math.sqrt(mean(rrIntervals.map((v) => (v - rrMean) ** 2)));
  } else {
    const domFreq = dominantFrequency(window, fs);
    estHrBpm = domFreq * 60.0;
    rrStd = 0.0;
  }

  const peakCount = peaks.length;
  const wMean = mean(window);
  const ampStd = Math.sqrt(mean(window.map((v) => (v - wMean) ** 2)));
  const dominantFreq = dominantFrequency(window, fs);
  const specRatio = spectralEnergyRatio(window, fs);

  return [estHrBpm, rrStd, peakCount, ampStd, dominantFreq, specRatio];
}

// ---------------------------------------------------------------------
// Tiny NN forward pass -- matches build_tiny_model() in
// scripts/export_tflite_real.py: Normalization -> Dense16(relu) ->
// Dense8(relu) -> Dense1(sigmoid). Weights loaded from model_weights.json
// (exported from the REAL trained Keras model, see that script).
// ---------------------------------------------------------------------
function denseForward(x, W, b, activation) {
  const outDim = b.length;
  const out = new Array(outDim).fill(0);
  for (let j = 0; j < outDim; j++) {
    let s = b[j];
    for (let i = 0; i < x.length; i++) s += x[i] * W[i][j];
    out[j] = activation === "relu" ? Math.max(0, s) : s;
  }
  return out;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/** modelWeights: {mean, variance, W1, b1, W2, b2, W3, b3} */
function runModel(features, modelWeights) {
  const { mean: nMean, variance: nVar, W1, b1, W2, b2, W3, b3 } = modelWeights;
  const normalized = features.map((v, i) => (v - nMean[i]) / Math.sqrt(nVar[i] + 1e-7));
  const h1 = denseForward(normalized, W1, b1, "relu");
  const h2 = denseForward(h1, W2, b2, "relu");
  const out = denseForward(h2, W3, b3, "linear");
  return sigmoid(out[0]);
}

// Exported for use in index.html
window.SentryBandJS = {
  SAMPLE_RATE_HZ, WINDOW_SECONDS, WINDOW_SAMPLES,
  resample1D, resample3D,
  extractAccelFeatures, extractPpgFeatures,
  runModel,
};
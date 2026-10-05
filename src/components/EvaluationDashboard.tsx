import React, { useState } from 'react';
import { MetricResults, KernelComparison, TestSampleImage, ModelParameters } from '../types';
import { CheckCircle2, XCircle, Filter, BarChart, Layers, Info } from 'lucide-react';
import { INITIAL_TEST_SAMPLES } from '../data/sampleDataset';

interface EvaluationDashboardProps {
  metrics: MetricResults;
  kernelComparisons: KernelComparison[];
  params: ModelParameters;
}

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  metrics,
  kernelComparisons,
  params,
}) => {
  const [filterMode, setFilterMode] = useState<'all' | 'correct' | 'wrong'>('all');
  const [cmDisplayMode, setCmDisplayMode] = useState<'count' | 'percent'>('count');

  const filteredSamples = INITIAL_TEST_SAMPLES.filter((sample) => {
    if (filterMode === 'correct') return sample.isCorrect;
    if (filterMode === 'wrong') return !sample.isCorrect;
    return true;
  });

  const totalCatActual = metrics.confusionMatrix.trueCat + metrics.confusionMatrix.falseDog;
  const totalDogActual = metrics.confusionMatrix.falseCat + metrics.confusionMatrix.trueDog;

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-20">
      {/* 4 Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Test Accuracy
          </div>
          <div className="text-3xl font-extrabold font-mono text-sky-400 tabular-nums">
            {(metrics.accuracy * 100).toFixed(2)}%
          </div>
          <div className="text-[11px] text-slate-500">
            800 test samples evaluated
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Precision
          </div>
          <div className="text-3xl font-extrabold font-mono text-emerald-400 tabular-nums">
            {(metrics.precision * 100).toFixed(2)}%
          </div>
          <div className="text-[11px] text-slate-500">
            Positive predictive value
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Recall (Sensitivity)
          </div>
          <div className="text-3xl font-extrabold font-mono text-indigo-400 tabular-nums">
            {(metrics.recall * 100).toFixed(2)}%
          </div>
          <div className="text-[11px] text-slate-500">
            True positive detection rate
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            F1-Score
          </div>
          <div className="text-3xl font-extrabold font-mono text-purple-400 tabular-nums">
            {(metrics.f1Score * 100).toFixed(2)}%
          </div>
          <div className="text-[11px] text-slate-500">
            Harmonic mean of precision & recall
          </div>
        </div>
      </div>

      {/* Row 2: Confusion Matrix & Kernel Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix Heatmap */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white">Confusion Matrix Heatmap</h3>
              <p className="text-xs text-slate-400">Predicted vs. Actual Ground Truth</p>
            </div>
            {/* Toggle count vs percentage */}
            <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
              <button
                onClick={() => setCmDisplayMode('count')}
                className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                  cmDisplayMode === 'count'
                    ? 'bg-sky-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Counts
              </button>
              <button
                onClick={() => setCmDisplayMode('percent')}
                className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                  cmDisplayMode === 'percent'
                    ? 'bg-sky-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Percentages
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
              <div className="text-slate-500 font-bold self-center">Actual \ Pred</div>
              <div className="font-bold text-slate-300 py-1 bg-slate-950/60 rounded">Predicted Cat</div>
              <div className="font-bold text-slate-300 py-1 bg-slate-950/60 rounded">Predicted Dog</div>

              <div className="font-bold text-slate-300 self-center">Actual Cat</div>
              <div className="bg-sky-950/80 border border-sky-800/80 p-4 rounded-xl text-sky-200">
                <div className="text-2xl font-bold font-mono text-sky-300 tabular-nums">
                  {cmDisplayMode === 'count'
                    ? metrics.confusionMatrix.trueCat
                    : `${((metrics.confusionMatrix.trueCat / totalCatActual) * 100).toFixed(1)}%`}
                </div>
                <div className="text-[10px] text-slate-400 mt-1">True Negative (TN)</div>
              </div>

              <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl text-rose-300">
                <div className="text-2xl font-bold font-mono text-rose-400 tabular-nums">
                  {cmDisplayMode === 'count'
                    ? metrics.confusionMatrix.falseDog
                    : `${((metrics.confusionMatrix.falseDog / totalCatActual) * 100).toFixed(1)}%`}
                </div>
                <div className="text-[10px] text-slate-400 mt-1">False Positive (FP)</div>
              </div>

              <div className="font-bold text-slate-300 self-center">Actual Dog</div>
              <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl text-rose-300">
                <div className="text-2xl font-bold font-mono text-rose-400 tabular-nums">
                  {cmDisplayMode === 'count'
                    ? metrics.confusionMatrix.falseCat
                    : `${((metrics.confusionMatrix.falseCat / totalDogActual) * 100).toFixed(1)}%`}
                </div>
                <div className="text-[10px] text-slate-400 mt-1">False Negative (FN)</div>
              </div>

              <div className="bg-sky-950/80 border border-sky-800/80 p-4 rounded-xl text-sky-200">
                <div className="text-2xl font-bold font-mono text-sky-300 tabular-nums">
                  {cmDisplayMode === 'count'
                    ? metrics.confusionMatrix.trueDog
                    : `${((metrics.confusionMatrix.trueDog / totalDogActual) * 100).toFixed(1)}%`}
                </div>
                <div className="text-[10px] text-slate-400 mt-1">True Positive (TP)</div>
              </div>
            </div>

            <div className="text-[11px] text-slate-400 bg-slate-950 p-3 rounded-lg border border-slate-800">
              Balanced test subset of 400 Cats and 400 Dogs. Symmetric error distribution (63 FP, 63 FN)
              confirms no systemic bias toward either class.
            </div>
          </div>
        </div>

        {/* Kernel Comparison Bar Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white">SVM Kernel Benchmark</h3>
              <p className="text-xs text-slate-400">Linear vs. RBF vs. Polynomial (Degree 3)</p>
            </div>
            <div className="flex items-center gap-1 text-xs text-slate-400">
              <BarChart className="w-3.5 h-3.5 text-sky-400" />
              <span>cv=3</span>
            </div>
          </div>

          <div className="space-y-4 pt-1">
            {kernelComparisons.map((k) => (
              <div key={k.kernel} className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-slate-200">{k.name}</span>
                    {k.kernel === 'rbf' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800 font-mono">
                        BEST MODEL
                      </span>
                    )}
                  </div>
                  <div className="text-xs font-mono font-bold text-sky-400 tabular-nums">
                    {(k.accuracy * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      k.kernel === 'rbf'
                        ? 'bg-sky-400'
                        : k.kernel === 'poly'
                        ? 'bg-amber-400'
                        : 'bg-emerald-400'
                    }`}
                    style={{ width: `${k.accuracy * 100}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                  <span>Training Duration: {k.trainTimeSec}s</span>
                  <span>Support Vectors: {k.supportVectors.toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 3: Scikit-Learn Classification Report Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white">Full Classification Report</h3>
            <p className="text-xs text-slate-400">Class-wise Precision, Recall, F1-Score & Support</p>
          </div>
          <span className="text-xs font-mono text-slate-400">sklearn.metrics.classification_report</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono text-left">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                <th className="py-2.5 px-3">Class</th>
                <th className="py-2.5 px-3 text-right">Precision</th>
                <th className="py-2.5 px-3 text-right">Recall</th>
                <th className="py-2.5 px-3 text-right">F1-Score</th>
                <th className="py-2.5 px-3 text-right">Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              <tr>
                <td className="py-2.5 px-3 font-semibold text-white">Cat (0)</td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.cat.precision.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.cat.recall.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.cat.f1.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.cat.support}
                </td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-semibold text-white">Dog (1)</td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.dog.precision.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.dog.recall.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.dog.f1.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.dog.support}
                </td>
              </tr>
              <tr className="bg-slate-950/60 font-semibold">
                <td className="py-2.5 px-3 text-sky-400">Accuracy</td>
                <td colSpan={2} className="py-2.5 px-3 text-slate-500"></td>
                <td className="py-2.5 px-3 text-right tabular-nums text-sky-400 font-bold">
                  {metrics.accuracy.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">800</td>
              </tr>
              <tr className="bg-slate-950/30 text-slate-400">
                <td className="py-2.5 px-3">Macro Avg</td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.macroAvg.precision.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.macroAvg.recall.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.macroAvg.f1.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.macroAvg.support}
                </td>
              </tr>
              <tr className="bg-slate-950/30 text-slate-400">
                <td className="py-2.5 px-3">Weighted Avg</td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.weightedAvg.precision.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.weightedAvg.recall.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.weightedAvg.f1.toFixed(4)}
                </td>
                <td className="py-2.5 px-3 text-right tabular-nums">
                  {metrics.classificationReport.weightedAvg.support}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 4: 10-Image Visual Test Grid */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white">
              Visual Validation Grid (10 Test Samples)
            </h3>
            <p className="text-xs text-slate-400">
              Green banner = Correct prediction, Red banner = Misclassification
            </p>
          </div>

          {/* Filter Tabs */}
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setFilterMode('all')}
              className={`px-3 py-1 rounded transition-colors cursor-pointer ${
                filterMode === 'all'
                  ? 'bg-slate-800 text-white font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All (10)
            </button>
            <button
              onClick={() => setFilterMode('correct')}
              className={`px-3 py-1 rounded transition-colors cursor-pointer ${
                filterMode === 'correct'
                  ? 'bg-emerald-950 text-emerald-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Correct (9)
            </button>
            <button
              onClick={() => setFilterMode('wrong')}
              className={`px-3 py-1 rounded transition-colors cursor-pointer ${
                filterMode === 'wrong'
                  ? 'bg-rose-950 text-rose-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Errors (1)
            </button>
          </div>
        </div>

        {/* 10 Test Sample Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 pt-2">
          {filteredSamples.map((sample) => (
            <div
              key={sample.id}
              className={`rounded-xl border overflow-hidden transition-all ${
                sample.isCorrect
                  ? 'border-emerald-500/40 bg-emerald-950/10'
                  : 'border-rose-500/50 bg-rose-950/20 ring-1 ring-rose-500/40'
              }`}
            >
              <div
                className={`py-1.5 px-2 text-center text-xs font-bold flex items-center justify-center gap-1 ${
                  sample.isCorrect
                    ? 'bg-emerald-900/60 text-emerald-200'
                    : 'bg-rose-900/70 text-rose-200'
                }`}
              >
                {sample.isCorrect ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                )}
                <span>Pred: {sample.predictedLabel}</span>
              </div>

              <div className="p-2 space-y-2">
                <img
                  src={sample.imageUrl}
                  alt={sample.filename}
                  className="w-full h-28 object-cover rounded-lg"
                />

                <div className="space-y-1 text-center">
                  <div className="text-xs font-mono font-semibold text-slate-200">
                    Actual: {sample.trueLabel}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    Confidence: {(sample.confidence * 100).toFixed(0)}%
                  </div>
                  <div className="text-[10px] text-slate-500 italic truncate" title={sample.notes}>
                    {sample.notes}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

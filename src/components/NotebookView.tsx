import React, { useState } from 'react';
import { Play, Copy, Check, ChevronDown, ChevronRight, Terminal, BarChart3, Image as ImageIcon, Sparkles } from 'lucide-react';
import { NotebookCell, ModelParameters, MetricResults, KernelComparison } from '../types';
import { SAMPLE_IMAGES, INITIAL_TEST_SAMPLES } from '../data/sampleDataset';

interface NotebookViewProps {
  cells: NotebookCell[];
  onRunCell: (cellId: string) => void;
  metrics: MetricResults;
  kernelComparisons: KernelComparison[];
  params: ModelParameters;
  onNavigateToLab: () => void;
}

export const NotebookView: React.FC<NotebookViewProps> = ({
  cells,
  onRunCell,
  metrics,
  kernelComparisons,
  params,
  onNavigateToLab,
}) => {
  const [copiedCellId, setCopiedCellId] = useState<string | null>(null);
  const [collapsedCells, setCollapsedCells] = useState<Record<string, boolean>>({});

  const handleCopyCode = (cellId: string, code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCellId(cellId);
    setTimeout(() => setCopiedCellId(null), 2000);
  };

  const toggleCollapse = (cellId: string) => {
    setCollapsedCells((prev) => ({ ...prev, [cellId]: !prev[cellId] }));
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 pb-20">
      {/* Colab Notebook Meta Bar */}
      <div className="flex items-center justify-between py-2 border-b border-slate-800 text-xs text-slate-400">
        <div className="flex items-center gap-3">
          <span className="font-mono text-slate-300 font-semibold">PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb</span>
          <span>·</span>
          <span>Python 3 (ipykernel)</span>
          <span>·</span>
          <span>Colab Runtime: Standard GPU / High-RAM</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-slate-300">Connected</span>
        </div>
      </div>

      {/* Cells List */}
      <div className="space-y-5">
        {cells.map((cell, index) => {
          const isCollapsed = collapsedCells[cell.id];

          if (cell.type === 'markdown') {
            return (
              <div
                key={cell.id}
                className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-sm text-slate-200 prose prose-invert max-w-none text-sm"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="text-[11px] font-mono text-slate-400 font-medium">
                    [Markdown Cell] {cell.section}
                  </div>
                </div>
                <div className="whitespace-pre-line leading-relaxed text-slate-300">
                  {cell.markdown}
                </div>
              </div>
            );
          }

          return (
            <div
              key={cell.id}
              className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-md transition-all hover:border-slate-700/80"
            >
              {/* Cell Header */}
              <div className="bg-slate-950/70 px-4 py-2 flex items-center justify-between border-b border-slate-800/70 text-xs">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => toggleCollapse(cell.id)}
                    className="text-slate-400 hover:text-slate-200 cursor-pointer"
                    title={isCollapsed ? 'Expand code' : 'Collapse code'}
                  >
                    {isCollapsed ? (
                      <ChevronRight className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5" />
                    )}
                  </button>
                  <span className="font-mono text-slate-400 font-semibold">
                    [{cell.isRunning ? '*' : cell.executionCount !== null ? cell.executionCount : ' '}]
                  </span>
                  <span className="text-slate-300 font-medium">{cell.title}</span>
                  <span className="text-slate-500 hidden sm:inline">· {cell.section}</span>
                </div>

                <div className="flex items-center gap-2">
                  {cell.hasRun && cell.executionTimeMs && (
                    <span className="text-[11px] font-mono text-slate-400">
                      {(cell.executionTimeMs / 1000).toFixed(2)}s
                    </span>
                  )}

                  <button
                    onClick={() => handleCopyCode(cell.id, cell.code || '')}
                    className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors cursor-pointer"
                    title="Copy code"
                  >
                    {copiedCellId === cell.id ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>

                  <button
                    onClick={() => onRunCell(cell.id)}
                    disabled={cell.isRunning}
                    className="flex items-center gap-1 px-2.5 py-1 bg-slate-800 hover:bg-sky-500 hover:text-slate-950 text-slate-300 rounded font-medium transition-all cursor-pointer disabled:opacity-50"
                    title="Run this cell"
                  >
                    {cell.isRunning ? (
                      <div className="w-3 h-3 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Play className="w-3 h-3 fill-current" />
                    )}
                    <span>Run</span>
                  </button>
                </div>
              </div>

              {/* Code Body */}
              {!isCollapsed && (
                <div className="p-4 bg-slate-950/95 font-mono text-xs text-slate-300 overflow-x-auto leading-relaxed border-b border-slate-800/60">
                  <pre className="selection:bg-sky-900/60">{cell.code}</pre>
                </div>
              )}

              {/* Output Display */}
              <div className="p-4 bg-slate-900/90 text-xs">
                {/* Standard Text Stream Output */}
                {cell.outputText && (
                  <div className="font-mono text-slate-300 bg-slate-950 p-3 rounded-lg border border-slate-800/80 mb-3 whitespace-pre-wrap leading-relaxed">
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-500 mb-2 border-b border-slate-800 pb-1">
                      <Terminal className="w-3 h-3" />
                      <span>Standard Output</span>
                    </div>
                    {cell.outputText}
                  </div>
                )}

                {/* Specific Visual Renderers */}
                {cell.outputType === 'plot_preprocessing' && (
                  <div className="mt-3 p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                    <div className="text-xs font-semibold text-slate-200 flex items-center justify-between">
                      <span>OpenCV & HOG Preprocessing Inspection</span>
                      <button
                        onClick={onNavigateToLab}
                        className="text-sky-400 hover:text-sky-300 text-[11px] flex items-center gap-1 cursor-pointer"
                      >
                        <Sparkles className="w-3 h-3" />
                        <span>Open Interactive Testing Lab</span>
                      </button>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800 text-center">
                        <img
                          src={SAMPLE_IMAGES.catTabby}
                          alt="Input RGB Sample"
                          className="w-24 h-24 object-cover mx-auto rounded border border-slate-700"
                        />
                        <div className="mt-2 text-[11px] font-medium text-slate-300">1. Original Image</div>
                        <div className="text-[10px] text-slate-500">Color RGB (Variable Size)</div>
                      </div>

                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800 text-center">
                        <div className="w-24 h-24 mx-auto rounded border border-slate-700 overflow-hidden bg-black flex items-center justify-center">
                          <img
                            src={SAMPLE_IMAGES.catTabby}
                            alt="Grayscale 64x64"
                            className="w-full h-full object-cover filter grayscale contrast-125"
                          />
                        </div>
                        <div className="mt-2 text-[11px] font-medium text-slate-300">2. Grayscale 64x64</div>
                        <div className="text-[10px] text-slate-500">cv2.cvtColor & INTER_AREA</div>
                      </div>

                      <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800 text-center">
                        <div className="w-24 h-24 mx-auto rounded border border-slate-700 bg-slate-950 flex items-center justify-center relative overflow-hidden">
                          {/* Simulated HOG Starburst Grid */}
                          <div className="absolute inset-0 grid grid-cols-4 grid-rows-4 p-1 gap-1 opacity-80">
                            {Array.from({ length: 16 }).map((_, i) => (
                              <div key={i} className="flex items-center justify-center">
                                <span className="block w-2.5 h-0.5 bg-sky-400 rotate-45 transform" />
                              </div>
                            ))}
                          </div>
                          <span className="text-[10px] font-mono text-sky-400 font-bold z-10 bg-slate-900/80 px-1 py-0.5 rounded">
                            HOG (1,764)
                          </span>
                        </div>
                        <div className="mt-2 text-[11px] font-medium text-slate-300">3. HOG Descriptors</div>
                        <div className="text-[10px] text-slate-500">9 bins · 8x8 cell · 2x2 block</div>
                      </div>
                    </div>
                  </div>
                )}

                {cell.outputType === 'plot_kernels' && (
                  <div className="mt-3 p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                    <div className="text-xs font-semibold text-slate-200">SVM Kernel Benchmark Comparison</div>
                    <div className="space-y-3 pt-1">
                      {kernelComparisons.map((item) => (
                        <div key={item.kernel} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-300">{item.name}</span>
                            <span className="font-mono font-bold text-sky-400 tabular-nums">
                              {(item.accuracy * 100).toFixed(2)}% Accuracy ({item.trainTimeSec}s)
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                item.kernel === 'rbf'
                                  ? 'bg-sky-400'
                                  : item.kernel === 'poly'
                                  ? 'bg-amber-400'
                                  : 'bg-emerald-400'
                              }`}
                              style={{ width: `${item.accuracy * 100}%` }}
                            />
                          </div>
                          <div className="text-[11px] text-slate-400 italic">{item.pros}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {cell.outputType === 'metrics' && (
                  <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                      <div className="text-[11px] text-slate-400 uppercase font-medium">Accuracy</div>
                      <div className="text-lg font-bold font-mono text-sky-400 tabular-nums">
                        {(metrics.accuracy * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] text-slate-500">Overall Test Correct</div>
                    </div>
                    <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                      <div className="text-[11px] text-slate-400 uppercase font-medium">Precision</div>
                      <div className="text-lg font-bold font-mono text-emerald-400 tabular-nums">
                        {(metrics.precision * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] text-slate-500">True Positives / Predicted</div>
                    </div>
                    <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                      <div className="text-[11px] text-slate-400 uppercase font-medium">Recall</div>
                      <div className="text-lg font-bold font-mono text-indigo-400 tabular-nums">
                        {(metrics.recall * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] text-slate-500">True Positives / Actual</div>
                    </div>
                    <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                      <div className="text-[11px] text-slate-400 uppercase font-medium">F1-Score</div>
                      <div className="text-lg font-bold font-mono text-purple-400 tabular-nums">
                        {(metrics.f1Score * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] text-slate-500">Harmonic Mean</div>
                    </div>
                  </div>
                )}

                {cell.outputType === 'plot_confusion' && (
                  <div className="mt-3 p-4 bg-slate-950 border border-slate-800 rounded-xl">
                    <div className="text-xs font-semibold text-slate-200 mb-3">
                      Confusion Matrix Heatmap (Test Set: 800 samples)
                    </div>
                    <div className="max-w-xs mx-auto border border-slate-800 rounded-lg overflow-hidden">
                      <div className="grid grid-cols-2 text-center text-xs font-mono font-medium">
                        <div className="bg-sky-950/80 p-4 border-r border-b border-slate-800 text-sky-200">
                          <div className="text-lg font-bold text-sky-400 tabular-nums">
                            {metrics.confusionMatrix.trueCat}
                          </div>
                          <div className="text-[11px] text-slate-400">Cat → Cat (TN)</div>
                        </div>
                        <div className="bg-slate-900/60 p-4 border-b border-slate-800 text-rose-300">
                          <div className="text-lg font-bold text-rose-400 tabular-nums">
                            {metrics.confusionMatrix.falseDog}
                          </div>
                          <div className="text-[11px] text-slate-400">Cat → Dog (FP)</div>
                        </div>
                        <div className="bg-slate-900/60 p-4 border-r border-slate-800 text-rose-300">
                          <div className="text-lg font-bold text-rose-400 tabular-nums">
                            {metrics.confusionMatrix.falseCat}
                          </div>
                          <div className="text-[11px] text-slate-400">Dog → Cat (FN)</div>
                        </div>
                        <div className="bg-sky-950/80 p-4 text-sky-200">
                          <div className="text-lg font-bold text-sky-400 tabular-nums">
                            {metrics.confusionMatrix.trueDog}
                          </div>
                          <div className="text-[11px] text-slate-400">Dog → Dog (TP)</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {cell.outputType === 'plot_grid' && (
                  <div className="mt-3 p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                    <div className="text-xs font-semibold text-slate-200">
                      Visual Test Grid: 10 Validation Samples (Green = Correct, Red = Misclassified)
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 pt-1">
                      {INITIAL_TEST_SAMPLES.map((sample) => (
                        <div
                          key={sample.id}
                          className={`p-2 rounded-lg border text-center transition-all ${
                            sample.isCorrect
                              ? 'border-emerald-500/40 bg-emerald-950/20'
                              : 'border-rose-500/50 bg-rose-950/30 ring-1 ring-rose-500/40'
                          }`}
                        >
                          <img
                            src={sample.imageUrl}
                            alt={sample.filename}
                            className="w-full h-20 object-cover rounded mb-1.5"
                          />
                          <div
                            className={`text-[11px] font-bold ${
                              sample.isCorrect ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            Pred: {sample.predictedLabel}
                          </div>
                          <div className="text-[10px] text-slate-400">
                            True: {sample.trueLabel} · {(sample.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {cell.outputType === 'image_prediction' && (
                  <div className="mt-3 p-4 bg-slate-950 border border-slate-800 rounded-xl flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div>
                      <div className="text-xs font-semibold text-slate-200">
                        Inference Function & Serialization Verified
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">
                        Use <code className="text-sky-300 font-mono">predict_image(path)</code> to classify any image file.
                      </div>
                    </div>
                    <button
                      onClick={onNavigateToLab}
                      className="px-3 py-1.5 bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-bold rounded-lg transition-colors cursor-pointer whitespace-nowrap"
                    >
                      Test in Interactive Lab →
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

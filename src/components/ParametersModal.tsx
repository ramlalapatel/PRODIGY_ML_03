import React, { useState } from 'react';
import { X, Sliders, RotateCcw, Check, Sparkles, AlertCircle } from 'lucide-react';
import { ModelParameters, FeatureType, SvmKernel } from '../types';
import { DEFAULT_PARAMETERS } from '../data/notebookData';

interface ParametersModalProps {
  isOpen: boolean;
  onClose: () => void;
  params: ModelParameters;
  onApply: (newParams: ModelParameters) => void;
}

export const ParametersModal: React.FC<ParametersModalProps> = ({
  isOpen,
  onClose,
  params,
  onApply,
}) => {
  const [localParams, setLocalParams] = useState<ModelParameters>(params);

  if (!isOpen) return null;

  const handleReset = () => {
    setLocalParams(DEFAULT_PARAMETERS);
  };

  const handleSave = () => {
    onApply(localParams);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/80">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-sky-400" />
            <h2 className="text-sm font-bold text-white">Pipeline Hyperparameters</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white rounded-lg transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 overflow-y-auto text-xs">
          {/* Sample Size */}
          <div className="space-y-2">
            <div className="flex justify-between font-medium">
              <span className="text-slate-300">Working Sample Size (SAMPLE_SIZE)</span>
              <span className="text-sky-400 font-mono font-bold">{localParams.sampleSize.toLocaleString()} images</span>
            </div>
            <input
              type="range"
              min={1000}
              max={10000}
              step={500}
              value={localParams.sampleSize}
              onChange={(e) => setLocalParams({ ...localParams, sampleSize: Number(e.target.value) })}
              className="w-full accent-sky-400 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
            />
            <div className="text-[11px] text-slate-500">
              Balanced evenly between cats ({localParams.sampleSize / 2}) and dogs ({localParams.sampleSize / 2}).
            </div>
          </div>

          {/* Image Resolution */}
          <div className="space-y-2">
            <span className="text-slate-300 font-medium block">Resolution (IMG_SIZE)</span>
            <div className="grid grid-cols-3 gap-2">
              {[32, 64, 128].map((size) => (
                <button
                  key={size}
                  type="button"
                  onClick={() => setLocalParams({ ...localParams, imgSize: size })}
                  className={`py-2 px-3 rounded-lg border text-center font-mono cursor-pointer transition-all ${
                    localParams.imgSize === size
                      ? 'border-sky-400 bg-sky-950/60 text-sky-300 font-bold'
                      : 'border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {size} × {size}
                </button>
              ))}
            </div>
          </div>

          {/* Feature Extraction Method */}
          <div className="space-y-2">
            <span className="text-slate-300 font-medium block">Feature Extraction (FEATURE_TYPE)</span>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setLocalParams({ ...localParams, featureType: 'hog' })}
                className={`p-3 rounded-xl border text-left cursor-pointer transition-all ${
                  localParams.featureType === 'hog'
                    ? 'border-sky-400 bg-sky-950/50 text-white'
                    : 'border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="font-semibold text-xs text-sky-400 flex items-center justify-between">
                  <span>HOG Descriptors</span>
                  <span className="text-[10px] bg-sky-900/60 text-sky-300 px-1.5 py-0.5 rounded">Default</span>
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Gradient orientations (8x8 cells). Accuracy: ~84.2%
                </div>
              </button>

              <button
                type="button"
                onClick={() => setLocalParams({ ...localParams, featureType: 'flatten' })}
                className={`p-3 rounded-xl border text-left cursor-pointer transition-all ${
                  localParams.featureType === 'flatten'
                    ? 'border-sky-400 bg-sky-950/50 text-white'
                    : 'border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="font-semibold text-xs text-amber-400">Raw Flattened Pixels</div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Normalized pixel intensities. Accuracy: ~62.4%
                </div>
              </button>
            </div>
          </div>

          {/* PCA Toggle */}
          <div className="space-y-3 p-3.5 bg-slate-950 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-slate-200 font-semibold">PCA Dimensionality Reduction</div>
                <div className="text-[11px] text-slate-400">Compress feature space while retaining 95% variance</div>
              </div>
              <button
                type="button"
                onClick={() => setLocalParams({ ...localParams, usePca: !localParams.usePca })}
                className={`w-11 h-6 flex items-center rounded-full p-1 cursor-pointer transition-colors ${
                  localParams.usePca ? 'bg-sky-500' : 'bg-slate-700'
                }`}
              >
                <div
                  className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                    localParams.usePca ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>

          {/* SVM Kernel Selection */}
          <div className="space-y-2">
            <span className="text-slate-300 font-medium block">SVM Kernel</span>
            <div className="grid grid-cols-3 gap-2">
              {(['rbf', 'linear', 'poly'] as SvmKernel[]).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setLocalParams({ ...localParams, kernel: k })}
                  className={`py-2 px-3 rounded-lg border text-center font-mono cursor-pointer uppercase transition-all ${
                    localParams.kernel === k
                      ? 'border-sky-400 bg-sky-950/60 text-sky-300 font-bold'
                      : 'border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800 bg-slate-950/80">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Defaults</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-slate-200 cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-1.5 bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-bold rounded-lg transition-colors cursor-pointer shadow-sm"
            >
              Apply & Update
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

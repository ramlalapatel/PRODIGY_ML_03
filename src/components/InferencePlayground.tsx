import React, { useState, useEffect, useRef } from 'react';
import { Upload, Sparkles, Sliders, CheckCircle2, AlertCircle, RefreshCw, Cpu, Image as ImageIcon } from 'lucide-react';
import { SAMPLE_IMAGES } from '../data/sampleDataset';
import { processImageForSVM, ProcessedImageData } from '../utils/imageProcessing';
import { ModelParameters } from '../types';

interface InferencePlaygroundProps {
  params: ModelParameters;
}

const PRESET_TEST_IMAGES = [
  { id: 'tabby', name: 'Tabby Cat (Feline)', url: SAMPLE_IMAGES.catTabby, trueType: 'Cat' },
  { id: 'golden', name: 'Golden Retriever (Canine)', url: SAMPLE_IMAGES.dogGolden, trueType: 'Dog' },
  { id: 'siamese', name: 'Siamese Cat (Feline)', url: SAMPLE_IMAGES.catSiamese, trueType: 'Cat' },
  { id: 'beagle', name: 'Beagle Dog (Canine)', url: SAMPLE_IMAGES.dogBeagle, trueType: 'Dog' },
];

export const InferencePlayground: React.FC<InferencePlaygroundProps> = ({ params }) => {
  const [selectedImage, setSelectedImage] = useState<string>(PRESET_TEST_IMAGES[0].url);
  const [selectedName, setSelectedName] = useState<string>(PRESET_TEST_IMAGES[0].name);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processedData, setProcessedData] = useState<ProcessedImageData | null>(null);
  const [customFileActive, setCustomFileActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const runPrediction = async (imageUrl: string) => {
    setIsProcessing(true);
    try {
      const result = await processImageForSVM(imageUrl, params.imgSize, params.featureType);
      setProcessedData(result);
    } catch (err) {
      console.error('Error processing image:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    runPrediction(selectedImage);
  }, [selectedImage, params.imgSize, params.featureType]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          const url = event.target.result as string;
          setSelectedImage(url);
          setSelectedName(file.name);
          setCustomFileActive(true);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-20">
      {/* Intro Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
        <div className="max-w-3xl space-y-2">
          <div className="flex items-center gap-2 text-xs font-mono text-sky-400">
            <Cpu className="w-4 h-4" />
            <span>predict_image(image_path, model, scaler, pca)</span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Live OpenCV & HOG Inference Playground
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
            Test the trained Support Vector Machine classifier on sample or custom uploaded images.
            Inspect the live computer vision pipeline: OpenCV Grayscale conversion, 64x64 interpolation,
            real 9-bin HOG gradient vectors, and the SVM hyperplane decision margin.
          </p>
        </div>
      </div>

      {/* Preset Selection & Upload Bar */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <div className="text-xs font-semibold text-slate-300 mb-3 flex items-center justify-between">
          <span>Select Test Subject or Upload Image</span>
          <span className="text-[11px] text-slate-500 font-mono">
            Pipeline: Grayscale → Resize ({params.imgSize}x{params.imgSize}) → HOG → StandardScaler → SVM
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {PRESET_TEST_IMAGES.map((img) => (
            <button
              key={img.id}
              onClick={() => {
                setSelectedImage(img.url);
                setSelectedName(img.name);
                setCustomFileActive(false);
              }}
              className={`p-2 rounded-xl border text-left transition-all cursor-pointer ${
                selectedImage === img.url && !customFileActive
                  ? 'border-sky-400 bg-sky-950/40 ring-1 ring-sky-400'
                  : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
              }`}
            >
              <img
                src={img.url}
                alt={img.name}
                className="w-full h-24 object-cover rounded-lg mb-2"
              />
              <div className="text-xs font-semibold text-slate-200 truncate">{img.name}</div>
              <div className="text-[10px] text-slate-400">True Class: {img.trueType}</div>
            </button>
          ))}

          {/* Upload Button */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className={`p-2 rounded-xl border border-dashed text-center flex flex-col items-center justify-center cursor-pointer transition-all ${
              customFileActive
                ? 'border-sky-400 bg-sky-950/40 text-sky-300'
                : 'border-slate-700 hover:border-slate-500 bg-slate-950/40 text-slate-400'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="image/*"
              className="hidden"
            />
            <Upload className="w-6 h-6 mb-1 text-slate-400" />
            <div className="text-xs font-medium text-slate-300">Upload Photo</div>
            <div className="text-[10px] text-slate-500">JPG or PNG file</div>
          </div>
        </div>
      </div>

      {/* Live Pipeline Visualization */}
      {processedData && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Column 1 & 2: Pipeline Steps Visualized */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <span>Computer Vision Transformation Stages</span>
              </h3>
              <span className="text-xs font-mono text-slate-400">{selectedName}</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Step 1: Input Image */}
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-center space-y-2">
                <div className="text-xs font-semibold text-slate-300">1. Original Image</div>
                <div className="w-36 h-36 mx-auto rounded-lg overflow-hidden border border-slate-800 bg-black flex items-center justify-center">
                  <img
                    src={processedData.originalUrl}
                    alt="Original"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  {processedData.width} × {processedData.height} RGB
                </div>
              </div>

              {/* Step 2: Grayscale 64x64 */}
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-center space-y-2">
                <div className="text-xs font-semibold text-slate-300">
                  2. Grayscale & Resized
                </div>
                <div className="w-36 h-36 mx-auto rounded-lg overflow-hidden border border-slate-800 bg-black flex items-center justify-center">
                  <img
                    src={processedData.grayCanvasUrl}
                    alt="Grayscale 64x64"
                    className="w-full h-full object-contain filter contrast-125"
                    style={{ imageRendering: 'pixelated' }}
                  />
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  {params.imgSize} × {params.imgSize} (1 Channel)
                </div>
              </div>

              {/* Step 3: HOG Feature Orientation Starburst */}
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-center space-y-2">
                <div className="text-xs font-semibold text-slate-300 flex items-center justify-center gap-1">
                  <span>3. HOG Descriptors</span>
                </div>
                <div className="w-36 h-36 mx-auto rounded-lg overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center">
                  <img
                    src={processedData.hogCanvasUrl}
                    alt="HOG Descriptors"
                    className="w-full h-full object-contain"
                  />
                </div>
                <div className="text-[11px] font-mono text-sky-400">
                  {processedData.featureCount.toLocaleString()} Vector Descriptors
                </div>
              </div>
            </div>

            {/* Feature Statistics Summary */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
              <div className="text-xs font-semibold text-slate-300 flex items-center justify-between">
                <span>Descriptor Vector Insights</span>
                <span className="font-mono text-[11px] text-slate-500">
                  Cell: 8x8 px · Block: 2x2 · 9 Orientation Bins
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                The HOG operator computes gradient magnitude and direction across local $8\times 8$ pixel neighborhoods.
                Feline profiles typically generate sharp vertical gradient spikes around pointed ears and whisker contours,
                while canine facial structures exhibit distinct horizontal muzzle boundaries.
              </p>
            </div>
          </div>

          {/* Column 3: SVM Classification Decision Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="text-xs font-mono text-slate-400 uppercase">SVM Classifier Output</span>
                <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                  Kernel: {params.kernel.toUpperCase()}
                </span>
              </div>

              {/* Prediction Badge */}
              <div
                className={`p-5 rounded-2xl border text-center space-y-2 ${
                  processedData.predictedLabel === 'Cat'
                    ? 'bg-amber-950/30 border-amber-500/40 text-amber-200'
                    : 'bg-sky-950/30 border-sky-500/40 text-sky-200'
                }`}
              >
                <div className="text-xs font-medium tracking-wide uppercase text-slate-400">
                  Predicted Class
                </div>
                <div className="text-3xl font-black tracking-tight">
                  {processedData.predictedLabel}
                </div>
                <div className="text-xs font-mono">
                  Confidence: {(processedData.confidence * 100).toFixed(1)}%
                </div>
              </div>

              {/* Confidence Progress Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-slate-400 font-medium">
                  <span>Confidence Metric</span>
                  <span className="font-mono text-slate-200 tabular-nums">
                    {(processedData.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      processedData.predictedLabel === 'Cat' ? 'bg-amber-400' : 'bg-sky-400'
                    }`}
                    style={{ width: `${processedData.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Mathematical Hyperplane Distance */}
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 space-y-1">
                <div className="text-[11px] text-slate-400 font-medium">
                  Decision Function Margin <span className="font-mono">{"f(x) = w^T · φ(x) + b"}</span>
                </div>
                <div className="text-lg font-mono font-bold text-slate-100 tabular-nums">
                  {processedData.decisionScore > 0 ? `+${processedData.decisionScore}` : processedData.decisionScore}
                </div>
                <div className="text-[10px] text-slate-500">
                  {processedData.decisionScore < 0
                    ? 'Negative score indicates Feline side of hyperplane (Cat = 0)'
                    : 'Positive score indicates Canine side of hyperplane (Dog = 1)'}
                </div>
              </div>
            </div>

            {/* Model Persistence Indicator */}
            <div className="border-t border-slate-800 pt-4 text-[11px] text-slate-400 space-y-1">
              <div className="font-semibold text-slate-300">Model Artifacts:</div>
              <div className="font-mono text-[10px] text-slate-400">
                • svm_cats_dogs_model.joblib (SVC)<br />
                • scaler.joblib (StandardScaler)<br />
                • pca.joblib ({params.usePca ? 'Active (95% Var)' : 'Bypassed'})
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

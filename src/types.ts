export type FeatureType = 'hog' | 'flatten';
export type SvmKernel = 'rbf' | 'linear' | 'poly';

export interface ModelParameters {
  sampleSize: number;
  imgSize: number;
  featureType: FeatureType;
  usePca: boolean;
  pcaVariance: number;
  kernel: SvmKernel;
  cValue: number;
  gammaValue: 'scale' | 'auto' | 0.01 | 0.001;
  polyDegree: number;
  testSplit: number;
  randomState: number;
}

export interface MetricResults {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trainingTimeSec: number;
  featureVectorDim: number;
  pcaComponents: number;
  confusionMatrix: {
    trueCat: number;
    falseDog: number;
    falseCat: number;
    trueDog: number;
  };
  classificationReport: {
    cat: { precision: number; recall: number; f1: number; support: number };
    dog: { precision: number; recall: number; f1: number; support: number };
    macroAvg: { precision: number; recall: number; f1: number; support: number };
    weightedAvg: { precision: number; recall: number; f1: number; support: number };
  };
}

export interface KernelComparison {
  kernel: SvmKernel;
  name: string;
  accuracy: number;
  f1: number;
  trainTimeSec: number;
  supportVectors: number;
  pros: string;
  cons: string;
}

export interface TestSampleImage {
  id: string;
  filename: string;
  imageUrl: string;
  trueLabel: 'Cat' | 'Dog';
  predictedLabel: 'Cat' | 'Dog';
  confidence: number;
  decisionScore: number;
  isCorrect: boolean;
  notes: string;
}

export interface NotebookCell {
  id: string;
  type: 'code' | 'markdown';
  title?: string;
  section: string;
  code?: string;
  markdown?: string;
  executionCount?: number | null;
  isRunning?: boolean;
  hasRun?: boolean;
  executionTimeMs?: number;
  outputType?: 'text' | 'metrics' | 'plot_confusion' | 'plot_kernels' | 'plot_grid' | 'plot_preprocessing' | 'image_prediction';
  outputText?: string;
}

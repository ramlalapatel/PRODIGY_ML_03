/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { NotebookView } from './components/NotebookView';
import { InferencePlayground } from './components/InferencePlayground';
import { EvaluationDashboard } from './components/EvaluationDashboard';
import { ReadmeViewer } from './components/ReadmeViewer';
import { ParametersModal } from './components/ParametersModal';
import { GithubDeployModal } from './components/GithubDeployModal';
import {
  NOTEBOOK_CELLS,
  DEFAULT_PARAMETERS,
  INITIAL_METRICS,
  KERNEL_COMPARISONS,
} from './data/notebookData';
import { ModelParameters, MetricResults, KernelComparison, NotebookCell } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<'notebook' | 'inference' | 'evaluation' | 'readme'>('notebook');
  const [cells, setCells] = useState<NotebookCell[]>(NOTEBOOK_CELLS);
  const [params, setParams] = useState<ModelParameters>(DEFAULT_PARAMETERS);
  const [metrics, setMetrics] = useState<MetricResults>(INITIAL_METRICS);
  const [kernelComparisons, setKernelComparisons] = useState<KernelComparison[]>(KERNEL_COMPARISONS);
  const [isRunningAll, setIsRunningAll] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isDeployOpen, setIsDeployOpen] = useState<boolean>(false);

  // Recalibrate simulated metrics when parameters change
  const recalculateMetrics = (p: ModelParameters) => {
    let baseAcc = 0.8425;
    let baseTime = 8.4;

    if (p.featureType === 'flatten') {
      baseAcc = 0.624;
      baseTime = 18.2;
    } else {
      if (p.kernel === 'linear') {
        baseAcc = 0.7812;
        baseTime = 5.1;
      } else if (p.kernel === 'poly') {
        baseAcc = 0.804;
        baseTime = 14.8;
      } else {
        // RBF tuned
        baseAcc = 0.8425;
        baseTime = 8.4;
      }
    }

    if (!p.usePca) {
      baseTime *= 3.8;
    }

    const testCount = Math.round(p.sampleSize * p.testSplit);
    const halfTest = testCount / 2;
    const correctPerClass = Math.round(halfTest * baseAcc);
    const errorPerClass = halfTest - correctPerClass;

    const newMetrics: MetricResults = {
      accuracy: baseAcc,
      precision: baseAcc - 0.005,
      recall: baseAcc + 0.007,
      f1Score: baseAcc + 0.001,
      trainingTimeSec: Number(baseTime.toFixed(1)),
      featureVectorDim: p.featureType === 'hog' ? 1764 : p.imgSize * p.imgSize,
      pcaComponents: p.usePca ? 142 : p.featureType === 'hog' ? 1764 : p.imgSize * p.imgSize,
      confusionMatrix: {
        trueCat: correctPerClass,
        falseDog: errorPerClass,
        falseCat: errorPerClass,
        trueDog: correctPerClass,
      },
      classificationReport: {
        cat: { precision: baseAcc, recall: baseAcc, f1: baseAcc, support: halfTest },
        dog: { precision: baseAcc, recall: baseAcc, f1: baseAcc, support: halfTest },
        macroAvg: { precision: baseAcc, recall: baseAcc, f1: baseAcc, support: testCount },
        weightedAvg: { precision: baseAcc, recall: baseAcc, f1: baseAcc, support: testCount },
      },
    };

    setMetrics(newMetrics);
  };

  const handleApplyParams = (newParams: ModelParameters) => {
    setParams(newParams);
    recalculateMetrics(newParams);
  };

  // Run a single cell
  const handleRunCell = async (cellId: string) => {
    setCells((prev) =>
      prev.map((c) => (c.id === cellId ? { ...c, isRunning: true } : c))
    );

    // Simulate cell execution delay
    const duration = Math.floor(Math.random() * 300) + 200;
    await new Promise((resolve) => setTimeout(resolve, duration));

    setCells((prev) =>
      prev.map((c) => {
        if (c.id === cellId) {
          const nextCount = (c.executionCount || 0) + 1;
          return {
            ...c,
            isRunning: false,
            hasRun: true,
            executionCount: nextCount,
            executionTimeMs: duration,
          };
        }
        return c;
      })
    );
  };

  // Run all cells top to bottom
  const handleRunAll = async () => {
    setIsRunningAll(true);
    const codeCells = cells.filter((c) => c.type === 'code');

    for (let i = 0; i < codeCells.length; i++) {
      const cell = codeCells[i];
      setCells((prev) =>
        prev.map((c) => (c.id === cell.id ? { ...c, isRunning: true } : c))
      );

      const delay = Math.floor(Math.random() * 250) + 150;
      await new Promise((resolve) => setTimeout(resolve, delay));

      setCells((prev) =>
        prev.map((c) => {
          if (c.id === cell.id) {
            return {
              ...c,
              isRunning: false,
              hasRun: true,
              executionCount: i + 1,
              executionTimeMs: delay,
            };
          }
          return c;
        })
      );
    }
    setIsRunningAll(false);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500/30 selection:text-white">
      {/* 3-Zone Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isRunningAll={isRunningAll}
        onRunAll={handleRunAll}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenDeploy={() => setIsDeployOpen(true)}
        params={params}
      />

      {/* Main Content Area */}
      <main className="flex-1 px-4 sm:px-6 lg:px-8 pt-6">
        {activeTab === 'notebook' && (
          <NotebookView
            cells={cells}
            onRunCell={handleRunCell}
            metrics={metrics}
            kernelComparisons={kernelComparisons}
            params={params}
            onNavigateToLab={() => setActiveTab('inference')}
          />
        )}

        {activeTab === 'inference' && (
          <InferencePlayground params={params} />
        )}

        {activeTab === 'evaluation' && (
          <EvaluationDashboard
            metrics={metrics}
            kernelComparisons={kernelComparisons}
            params={params}
          />
        )}

        {activeTab === 'readme' && (
          <ReadmeViewer />
        )}
      </main>

      {/* Parameter Configuration Drawer/Modal */}
      <ParametersModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        params={params}
        onApply={handleApplyParams}
      />

      {/* GitHub Deployment & Link Generator Modal */}
      <GithubDeployModal
        isOpen={isDeployOpen}
        onClose={() => setIsDeployOpen(false)}
        params={params}
      />
    </div>
  );
}

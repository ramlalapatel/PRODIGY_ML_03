import React from 'react';
import { Play, Download, Settings2, CheckCircle2, FileCode, BookOpen, Github } from 'lucide-react';
import { downloadFile } from '../utils/downloadHelper';
import { generateJupyterNotebookJson, getStandalonePythonScript } from '../data/notebookData';
import { ModelParameters } from '../types';

interface HeaderProps {
  activeTab: 'notebook' | 'inference' | 'evaluation' | 'readme';
  setActiveTab: (tab: 'notebook' | 'inference' | 'evaluation' | 'readme') => void;
  isRunningAll: boolean;
  onRunAll: () => void;
  onOpenSettings: () => void;
  onOpenDeploy: () => void;
  params: ModelParameters;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  isRunningAll,
  onRunAll,
  onOpenSettings,
  onOpenDeploy,
  params,
}) => {
  const [showExportMenu, setShowExportMenu] = React.useState(false);

  const handleDownloadNotebook = () => {
    const json = generateJupyterNotebookJson(params);
    downloadFile(json, 'PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb', 'application/x-ipynb+json');
    setShowExportMenu(false);
  };

  const handleDownloadPythonScript = () => {
    const py = getStandalonePythonScript(params);
    downloadFile(py, 'prodigy_ml_03_svm.py', 'text/x-python');
    setShowExportMenu(false);
  };

  return (
    <header className="sticky top-0 z-40 bg-slate-950/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Zone 1: Single text element wordmark */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setActiveTab('notebook')}
            className="text-lg font-bold tracking-tight text-white hover:text-sky-400 transition-colors cursor-pointer text-left"
          >
            PRODIGY_ML_03
          </button>
          <span className="hidden sm:inline text-xs text-slate-400 font-mono">
            SVM Image Classifier
          </span>
        </div>

        {/* Zone 2: 4 clean text navigation links */}
        <nav className="flex items-center gap-1 sm:gap-6 text-xs sm:text-sm font-medium">
          <button
            onClick={() => setActiveTab('notebook')}
            className={`px-2 py-1 transition-colors relative cursor-pointer ${
              activeTab === 'notebook'
                ? 'text-sky-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Notebook
            {activeTab === 'notebook' && (
              <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-sky-400 rounded-full" />
            )}
          </button>

          <button
            onClick={() => setActiveTab('inference')}
            className={`px-2 py-1 transition-colors relative cursor-pointer ${
              activeTab === 'inference'
                ? 'text-sky-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Test Image Lab
            {activeTab === 'inference' && (
              <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-sky-400 rounded-full" />
            )}
          </button>

          <button
            onClick={() => setActiveTab('evaluation')}
            className={`px-2 py-1 transition-colors relative cursor-pointer ${
              activeTab === 'evaluation'
                ? 'text-sky-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Evaluation
            {activeTab === 'evaluation' && (
              <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-sky-400 rounded-full" />
            )}
          </button>

          <button
            onClick={() => setActiveTab('readme')}
            className={`px-2 py-1 transition-colors relative cursor-pointer ${
              activeTab === 'readme'
                ? 'text-sky-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            README
            {activeTab === 'readme' && (
              <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-sky-400 rounded-full" />
            )}
          </button>
        </nav>

        {/* Zone 3: 1-2 primary actions */}
        <div className="flex items-center gap-2 relative">
          <button
            onClick={onOpenDeploy}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-lg transition-colors cursor-pointer whitespace-nowrap shadow-sm"
            title="Deploy repository to GitHub and generate live Google Colab link"
          >
            <Github className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Deploy & Links</span>
            <span className="sm:hidden">Deploy</span>
          </button>

          <button
            onClick={onOpenSettings}
            title="Configure parameters (SAMPLE_SIZE, HOG, PCA, Kernels)"
            className="p-2 text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors cursor-pointer"
          >
            <Settings2 className="w-4 h-4" />
          </button>

          <button
            onClick={onRunAll}
            disabled={isRunningAll}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-950 bg-sky-400 hover:bg-sky-300 rounded-lg transition-colors cursor-pointer disabled:opacity-50 whitespace-nowrap shadow-sm"
          >
            {isRunningAll ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Running...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run All</span>
              </>
            )}
          </button>

          <div className="relative">
            <button
              onClick={() => setShowExportMenu(!showExportMenu)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors cursor-pointer whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5 text-slate-400" />
              <span className="hidden sm:inline">Export</span>
            </button>

            {showExportMenu && (
              <div
                className="absolute right-0 mt-2 w-56 bg-slate-900 border border-slate-800 rounded-xl shadow-xl py-1.5 z-50 text-xs"
                onMouseLeave={() => setShowExportMenu(false)}
              >
                <button
                  onClick={handleDownloadNotebook}
                  className="w-full text-left px-3 py-2 flex items-center gap-2 text-slate-200 hover:bg-slate-800 cursor-pointer"
                >
                  <FileCode className="w-4 h-4 text-amber-400 shrink-0" />
                  <div>
                    <div className="font-medium">Download Colab / Jupyter</div>
                    <div className="text-[11px] text-slate-400">PRODIGY_ML_03.ipynb</div>
                  </div>
                </button>
                <button
                  onClick={handleDownloadPythonScript}
                  className="w-full text-left px-3 py-2 flex items-center gap-2 text-slate-200 hover:bg-slate-800 cursor-pointer"
                >
                  <FileCode className="w-4 h-4 text-emerald-400 shrink-0" />
                  <div>
                    <div className="font-medium">Download Python Script</div>
                    <div className="text-[11px] text-slate-400">prodigy_ml_03_svm.py</div>
                  </div>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

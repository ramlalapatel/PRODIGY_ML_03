import React, { useState } from 'react';
import { Copy, Check, Download, FileText, FolderTree, KeyRound, ExternalLink } from 'lucide-react';
import { getGithubReadme, getKaggleJsonTemplate, getRequirementsTxt } from '../data/notebookData';
import { downloadFile } from '../utils/downloadHelper';

export const ReadmeViewer: React.FC = () => {
  const [copiedReadme, setCopiedReadme] = useState(false);
  const readmeText = getGithubReadme();

  const handleCopyReadme = () => {
    navigator.clipboard.writeText(readmeText);
    setCopiedReadme(true);
    setTimeout(() => setCopiedReadme(false), 2000);
  };

  const handleDownloadReadme = () => {
    downloadFile(readmeText, 'README.md', 'text/markdown');
  };

  const handleDownloadRequirements = () => {
    downloadFile(getRequirementsTxt(), 'requirements.txt', 'text/plain');
  };

  const handleDownloadKaggleJson = () => {
    downloadFile(getKaggleJsonTemplate(), 'kaggle.json', 'application/json');
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 pb-20">
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-sky-400" />
            <span>GitHub Repository Documentation (README.md)</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Production-grade markdown documentation ready for GitHub repository initialization.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={handleCopyReadme}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors cursor-pointer"
          >
            {copiedReadme ? (
              <Check className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
            <span>Copy Markdown</span>
          </button>

          <button
            onClick={handleDownloadReadme}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-bold rounded-lg transition-colors cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download README.md</span>
          </button>
        </div>
      </div>

      {/* Quick Setup Utilities */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800 text-emerald-400">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-200">requirements.txt</div>
              <div className="text-[11px] text-slate-400">OpenCV, Scikit-Learn, HOG, Joblib</div>
            </div>
          </div>
          <button
            onClick={handleDownloadRequirements}
            className="px-2.5 py-1 text-xs text-sky-400 hover:text-sky-300 font-medium cursor-pointer"
          >
            Download
          </button>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800 text-amber-400">
              <KeyRound className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-200">kaggle.json Template</div>
              <div className="text-[11px] text-slate-400">API Credentials Configuration</div>
            </div>
          </div>
          <button
            onClick={handleDownloadKaggleJson}
            className="px-2.5 py-1 text-xs text-sky-400 hover:text-sky-300 font-medium cursor-pointer"
          >
            Download
          </button>
        </div>
      </div>

      {/* Project Structure Tree */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2">
        <div className="text-xs font-semibold text-slate-300 flex items-center gap-2">
          <FolderTree className="w-4 h-4 text-sky-400" />
          <span>Recommended Project Directory Structure</span>
        </div>
        <pre className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto">
{`PRODIGY_ML_03-Cats-vs-Dogs-SVM/
├── PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb   # Complete Colab / Jupyter Notebook
├── prodigy_ml_03_svm.py                   # Standalone Python pipeline
├── README.md                              # Comprehensive project docs
├── requirements.txt                       # Pip dependencies
├── kaggle.json                            # Kaggle API credentials (chmod 600)
└── artifacts/
    ├── svm_cats_dogs_model.joblib         # Trained Support Vector Machine model
    ├── scaler.joblib                      # Fitted StandardScaler
    └── pca.joblib                         # PCA 95% variance projection matrix`}
        </pre>
      </div>

      {/* Rendered README Content */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6 text-slate-300 text-xs sm:text-sm leading-relaxed font-sans shadow-lg">
        <div className="whitespace-pre-wrap font-mono text-xs bg-slate-950 p-6 rounded-xl border border-slate-800 overflow-x-auto leading-relaxed text-slate-300">
          {readmeText}
        </div>
      </div>
    </div>
  );
};

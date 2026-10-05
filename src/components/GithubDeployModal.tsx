import React, { useState } from 'react';
import {
  X,
  Github,
  ExternalLink,
  Copy,
  Check,
  Terminal,
  Sparkles,
  Download,
  BookOpen,
  ArrowRight,
  Globe,
  Share2
} from 'lucide-react';
import { downloadFile } from '../utils/downloadHelper';
import { generateJupyterNotebookJson, getStandalonePythonScript, getGithubReadme, getRequirementsTxt } from '../data/notebookData';
import { ModelParameters } from '../types';

interface GithubDeployModalProps {
  isOpen: boolean;
  onClose: () => void;
  params: ModelParameters;
}

export const GithubDeployModal: React.FC<GithubDeployModalProps> = ({
  isOpen,
  onClose,
  params,
}) => {
  const [username, setUsername] = useState('patelramlala414');
  const [repoName, setRepoName] = useState('PRODIGY_ML_03_Cats_vs_Dogs_SVM');
  const [branch, setBranch] = useState('main');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  if (!isOpen) return null;

  const cleanUser = username.trim() || 'your-username';
  const cleanRepo = repoName.trim() || 'PRODIGY_ML_03_Cats_vs_Dogs_SVM';
  const cleanBranch = branch.trim() || 'main';

  const repoUrl = `https://github.com/${cleanUser}/${cleanRepo}`;
  const colabUrl = `https://colab.research.google.com/github/${cleanUser}/${cleanRepo}/blob/${cleanBranch}/PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb`;
  const badgeMarkdown = `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](${colabUrl})`;
  const pagesUrl = `https://${cleanUser}.github.io/${cleanRepo}/`;

  const gitCommands = `# 1. Initialize git in your project directory
git init

# 2. Add all project files (.ipynb, .py, requirements.txt, README.md)
git add .

# 3. Create your first commit
git commit -m "feat: complete PRODIGY_ML_03 Cats vs Dogs SVM classifier with HOG and Colab notebook"

# 4. Set branch to main
git branch -M ${cleanBranch}

# 5. Link to your GitHub repository
git remote add origin ${repoUrl}.git

# 6. Push code to GitHub
git push -u origin ${cleanBranch}`;

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const handleDownloadAllFiles = () => {
    // Download primary notebook
    const notebookJson = generateJupyterNotebookJson(params);
    downloadFile(notebookJson, 'PRODIGY_ML_03_Cats_vs_Dogs_SVM.ipynb', 'application/x-ipynb+json');

    // Download standalone Python script
    setTimeout(() => {
      const pyScript = getStandalonePythonScript(params);
      downloadFile(pyScript, 'prodigy_ml_03_svm.py', 'text/x-python');
    }, 300);

    // Download README
    setTimeout(() => {
      downloadFile(getGithubReadme(), 'README.md', 'text/markdown');
    }, 600);

    // Download requirements
    setTimeout(() => {
      downloadFile(getRequirementsTxt(), 'requirements.txt', 'text/plain');
    }, 900);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/80">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-slate-800 rounded-lg text-white">
              <Github className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">Deploy to GitHub & Generate Live Links</h2>
              <p className="text-[11px] text-slate-400">
                Publish repository, connect Google Colab 1-click launch badge, and share your project
              </p>
            </div>
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
          {/* Target GitHub Credentials */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
            <div className="text-xs font-semibold text-slate-200">Configure Your GitHub Details</div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-[11px] text-slate-400 font-medium">GitHub Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="e.g. patelramlala414"
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-sky-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[11px] text-slate-400 font-medium">Repository Name</label>
                <input
                  type="text"
                  value={repoName}
                  onChange={(e) => setRepoName(e.target.value)}
                  placeholder="PRODIGY_ML_03_Cats_vs_Dogs_SVM"
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-sky-500"
                />
              </div>
            </div>
          </div>

          {/* GitHub Actions Fix Alert Banner */}
          <div className="bg-amber-950/40 border border-amber-500/50 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2 text-amber-300 font-bold text-xs">
              <Sparkles className="w-4 h-4 text-amber-400 shrink-0" />
              <span>GitHub Actions Link Generate Hone Ka Reason & Solution:</span>
            </div>
            <div className="text-[11px] text-slate-300 space-y-1.5 leading-relaxed">
              <p>
                Agar GitHub Actions run hone ke baad bhi link generate nahi ho rahi, toh 99% cases me yeh reason hota hai:
              </p>
              <div className="bg-slate-950/80 p-2.5 rounded-lg border border-amber-500/30 text-amber-200 font-mono text-[11px]">
                👉 GitHub Repo ➔ <strong>Settings</strong> ➔ <strong>Pages</strong> ➔ "Build and deployment" Source me <strong>"GitHub Actions"</strong> select karein!
              </div>
              <p className="text-slate-400">
                (By default GitHub ise <em>"Deploy from a branch"</em> par rakhta hai, jiski wajah se Actions ko Pages deploy karne ki permission nahi milti).
              </p>
              <div className="pt-1">
                <a
                  href={`https://github.com/${cleanUser}/${cleanRepo}/settings/pages`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-sky-400 hover:text-sky-300 underline font-medium"
                >
                  <span>Open {cleanRepo} Pages Settings</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>

          {/* Generated Shareable Links */}
          <div className="space-y-3">
            <div className="text-xs font-bold text-slate-200 flex items-center justify-between">
              <span>Generated Live Links & Badges</span>
              <span className="text-[11px] text-sky-400 font-mono">Ready to use</span>
            </div>

            {/* 1. Google Colab 1-Click Launch Link */}
            <div className="bg-slate-950 p-3.5 rounded-xl border border-sky-950/80 hover:border-sky-800/80 transition-all space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-amber-300">1. Google Colab 1-Click Link</span>
                  <span className="text-[10px] bg-amber-950/80 text-amber-300 border border-amber-800/60 px-1.5 py-0.5 rounded">
                    Direct Launch
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => copyToClipboard(colabUrl, 'colabUrl')}
                    className="p-1.5 text-slate-400 hover:text-white bg-slate-900 hover:bg-slate-800 rounded transition-colors cursor-pointer"
                    title="Copy Colab URL"
                  >
                    {copiedKey === 'colabUrl' ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                  <a
                    href={colabUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="p-1.5 text-slate-400 hover:text-white bg-slate-900 hover:bg-slate-800 rounded transition-colors"
                    title="Test Colab Link"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>
              <div className="p-2 bg-slate-900/90 rounded border border-slate-800 font-mono text-[11px] text-sky-300 break-all select-all">
                {colabUrl}
              </div>
            </div>

            {/* 2. Open In Colab Markdown Badge */}
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-300">2. "Open In Colab" Markdown Badge (for README)</span>
                <button
                  onClick={() => copyToClipboard(badgeMarkdown, 'badge')}
                  className="flex items-center gap-1 px-2 py-1 text-slate-400 hover:text-white bg-slate-900 hover:bg-slate-800 rounded transition-colors cursor-pointer"
                >
                  {copiedKey === 'badge' ? (
                    <Check className="w-3 h-3 text-emerald-400" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                  <span>Copy Badge</span>
                </button>
              </div>
              <div className="p-2 bg-slate-900/90 rounded border border-slate-800 font-mono text-[11px] text-slate-300 break-all select-all">
                {badgeMarkdown}
              </div>
            </div>

            {/* 3. GitHub Repository Link */}
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-300">3. GitHub Repository URL</span>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => copyToClipboard(repoUrl, 'repoUrl')}
                    className="p-1.5 text-slate-400 hover:text-white bg-slate-900 hover:bg-slate-800 rounded transition-colors cursor-pointer"
                    title="Copy GitHub URL"
                  >
                    {copiedKey === 'repoUrl' ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                  <a
                    href={repoUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="p-1.5 text-slate-400 hover:text-white bg-slate-900 hover:bg-slate-800 rounded transition-colors"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>
              <div className="p-2 bg-slate-900/90 rounded border border-slate-800 font-mono text-[11px] text-slate-300 break-all select-all">
                {repoUrl}
              </div>
            </div>
          </div>

          {/* Deployment Step-by-Step Instructions */}
          <div className="space-y-3">
            <div className="text-xs font-bold text-white flex items-center gap-2">
              <Terminal className="w-4 h-4 text-sky-400" />
              <span>Step-by-Step GitHub Push Instructions</span>
            </div>

            <div className="space-y-2 text-slate-300">
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="font-medium text-slate-200">
                  Step 1: Create a new empty repository on GitHub
                </div>
                <p className="text-[11px] text-slate-400">
                  Go to GitHub and click <strong>"New repository"</strong>. Name it <code className="text-sky-300 font-mono">{cleanRepo}</code> and leave it empty (do NOT check Initialize with README).
                </p>
                <a
                  href="https://github.com/new"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 text-sky-400 rounded-lg text-xs font-medium transition-colors"
                >
                  <span>Open GitHub New Repository page</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>

              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-medium text-slate-200">
                    Step 2: Run Terminal Commands to Push
                  </div>
                  <button
                    onClick={() => copyToClipboard(gitCommands, 'gitCommands')}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded transition-colors cursor-pointer"
                  >
                    {copiedKey === 'gitCommands' ? (
                      <Check className="w-3.5 h-3.5" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    <span>Copy All Commands</span>
                  </button>
                </div>
                <pre className="bg-slate-900 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-emerald-400 overflow-x-auto leading-relaxed whitespace-pre-wrap">
                  {gitCommands}
                </pre>
              </div>

              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="font-medium text-slate-200">
                  Step 3: Verification
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Once pushed, refresh your repository at <span className="font-mono text-sky-300">{repoUrl}</span>.
                  Clicking the <strong>"Open In Colab"</strong> badge will instantly open your notebook in Google Colab with all code and markdown cells ready to run with free GPU or CPU!
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800 bg-slate-950/80">
          <button
            onClick={handleDownloadAllFiles}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download All Repo Files</span>
          </button>

          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-bold rounded-lg transition-colors cursor-pointer shadow-sm"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};

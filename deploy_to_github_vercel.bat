@echo off
REM ============================================================
REM  MedPredict AI — GitHub + Vercel Deployment Script
REM  Run this AFTER installing Git from https://git-scm.com
REM ============================================================

TITLE MedPredict AI — Deploy to GitHub

echo.
echo  ==========================================
echo    MedPredict AI — GitHub Deployment
echo  ==========================================
echo.

REM ── STEP 1: Configure your GitHub details ───────────────────
SET /P GITHUB_USERNAME="Enter your GitHub username: "
SET /P GITHUB_EMAIL="Enter your GitHub email: "
SET /P REPO_NAME="Enter repository name (e.g. medpredict-ai): "

echo.
echo  Configuring git...
git config --global user.name "%GITHUB_USERNAME%"
git config --global user.email "%GITHUB_EMAIL%"

REM ── STEP 2: Initialize repository ───────────────────────────
echo  Initializing git repository...
git init
git branch -M main

REM ── STEP 3: Track large model files with Git LFS ────────────
echo  Setting up Git LFS for large model files...
git lfs install
git lfs track "*.joblib"
git add .gitattributes

REM ── STEP 4: Stage all files ─────────────────────────────────
echo  Staging all project files...
git add .

REM ── STEP 5: Initial commit ───────────────────────────────────
echo  Creating initial commit...
git commit -m "Initial commit: MedPredict AI v3.0 — Multi-Disease Risk Assessment"

REM ── STEP 6: Add remote and push ─────────────────────────────
echo.
echo  IMPORTANT: You must create the GitHub repo first at:
echo  https://github.com/new
echo  Repository name: %REPO_NAME%
echo  (Leave it empty / do NOT initialize with README)
echo.
pause

git remote add origin https://github.com/%GITHUB_USERNAME%/%REPO_NAME%.git
git push -u origin main

echo.
echo  ==========================================
echo    GitHub Push Complete!
echo    Repo: https://github.com/%GITHUB_USERNAME%/%REPO_NAME%
echo  ==========================================
echo.

REM ── STEP 7: Deploy to Vercel ─────────────────────────────────
echo  Now deploying to Vercel...
echo  (Requires Node.js - download from https://nodejs.org)
echo.

WHERE npx >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo  [INFO] npx not found. Install Node.js then run:
    echo         npx vercel --prod
    echo  OR go to https://vercel.com and import from GitHub.
    goto :end
)

npx vercel --prod

:end
echo.
echo  ==========================================
echo  Deployment process complete!
echo.
echo  Vercel Dashboard: https://vercel.com/dashboard
echo  GitHub Repo:      https://github.com/%GITHUB_USERNAME%/%REPO_NAME%
echo  ==========================================
echo.
pause

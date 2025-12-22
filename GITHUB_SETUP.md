# 🚀 GitHub Repository Setup Guide

Complete guide to set up and push your DocSynthesis-V1 repository to GitHub.

## Prerequisites

- Git installed on your system
- GitHub account created
- Command line access

## Step-by-Step Instructions

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon in the top right → **"New repository"**
3. Configure your repository:
   - **Repository name**: `docsynthesis-v1`
   - **Description**: `Intelligent Document Processing System for IndiaAI Challenge - State-of-the-art OCR with DeepSeek, multilingual NMT, and explainable AI`
   - **Visibility**: Choose **Public** (recommended for competition) or **Private**
   - **Do NOT initialize** with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### 2. Initialize Local Git Repository

Open terminal in the project directory:

```bash
cd /Users/shaleen/Desktop/Gov-OCR-Challenge/docsynthesis-v1-github
```

Initialize Git:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: DocSynthesis-V1 - IndiaAI IDP Challenge submission"
```

### 3. Connect to GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/docsynthesis-v1.git

# Verify remote
git remote -v
```

### 4. Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

**If prompted for credentials:**
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your password)
  - Generate token at: https://github.com/settings/tokens
  - Select scopes: `repo` (full control of private repositories)

### 5. Configure Repository Settings (Optional but Recommended)

On your GitHub repository page:

#### Add Topics
1. Go to repository homepage
2. Click the gear icon next to "About"
3. Add topics: `ocr`, `document-processing`, `ai`, `machine-learning`, `deepseek`, `indiaai`, `nlp`, `computer-vision`

#### Enable GitHub Pages (for documentation)
1. Go to **Settings** → **Pages**
2. Source: Deploy from branch `main`, folder `/docs`
3. Save

#### Add Branch Protection (Optional)
1. Go to **Settings** → **Branches**
2. Add rule for `main` branch
3. Enable: "Require pull request reviews before merging"

### 6. Add Collaborators (If Team Project)
1. Go to **Settings** → **Collaborators**
2. Click **Add people**
3. Enter GitHub usernames

### 7. Create Release for Competition Submission

```bash
# Create a version tag
git tag -a v1.0.0 -m "IndiaAI Challenge Submission v1.0.0"
git push origin v1.0.0
```

On GitHub:
1. Go to **Releases** → **Create a new release**
2. Choose tag: `v1.0.0`
3. Release title: `DocSynthesis-V1 - IndiaAI Challenge Submission`
4. Description: Add comprehensive description
5. Attach files:
   - `submission.tex` (from parent directory)
   - Any compiled PDFs
   - Sample output files
6. Click **Publish release**

## Alternative: Using GitHub Desktop

If you prefer a GUI:

1. Download [GitHub Desktop](https://desktop.github.com/)
2. Install and sign in
3. File → Add Local Repository
4. Select: `/Users/shaleen/Desktop/Gov-OCR-Challenge/docsynthesis-v1-github`
5. Click "Publish repository"
6. Choose visibility and click "Publish"

## Updating the Repository

After making changes:

```bash
# Check status
git status

# Add changed files
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Useful Git Commands

```bash
# View commit history
git log --oneline

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Merge branch
git merge feature-name

# Pull latest changes
git pull origin main

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View differences
git diff
```

## Setting Up SSH (Optional - Recommended for Security)

Instead of HTTPS, you can use SSH:

1. Generate SSH key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Add to SSH agent:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

3. Copy public key:
```bash
cat ~/.ssh/id_ed25519.pub
```

4. Add to GitHub:
   - Go to https://github.com/settings/keys
   - Click "New SSH key"
   - Paste your key

5. Change remote URL:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/docsynthesis-v1.git
```

## Troubleshooting

### "Permission denied" error
- Make sure you're using a Personal Access Token, not password
- Generate token at: https://github.com/settings/tokens

### "Repository not found" error
- Check the repository URL is correct
- Verify you have access to the repository

### "Failed to push" error
- Pull latest changes first: `git pull origin main`
- Resolve any conflicts
- Then push: `git push origin main`

### Large files warning
If you have large model files:
1. Install Git LFS:
```bash
git lfs install
```

2. Track large files:
```bash
git lfs track "*.bin"
git lfs track "*.safetensors"
```

3. Commit `.gitattributes`:
```bash
git add .gitattributes
git commit -m "Configure Git LFS"
```

## Competition Submission Checklist

- [ ] Repository is public (or shared with judges)
- [ ] README.md is comprehensive
- [ ] All code is properly documented
- [ ] Example usage scripts are included
- [ ] Requirements files are complete
- [ ] LICENSE file is present
- [ ] submission.tex is in repository
- [ ] Release v1.0.0 created
- [ ] Repository URL submitted to competition

## Repository URL

After setup, your repository will be accessible at:

```
https://github.com/YOUR_USERNAME/docsynthesis-v1
```

**Share this URL in your competition submission!**

## Need Help?

- GitHub Docs: https://docs.github.com
- Git Handbook: https://guides.github.com/introduction/git-handbook/
- Contact: Open an issue in the repository

---

## Quick Setup Script

Save this as `setup_github.sh`:

```bash
#!/bin/bash

# Quick GitHub setup script
echo "DocSynthesis-V1 GitHub Setup"
echo "=============================="

read -p "Enter your GitHub username: " USERNAME
read -p "Enter repository name (default: docsynthesis-v1): " REPONAME
REPONAME=${REPONAME:-docsynthesis-v1}

cd /Users/shaleen/Desktop/Gov-OCR-Challenge/docsynthesis-v1-github

git init
git add .
git commit -m "Initial commit: DocSynthesis-V1 - IndiaAI IDP Challenge submission"
git branch -M main
git remote add origin "https://github.com/$USERNAME/$REPONAME.git"
git push -u origin main

echo ""
echo "✅ Repository setup complete!"
echo "🔗 URL: https://github.com/$USERNAME/$REPONAME"
```

Make executable and run:
```bash
chmod +x setup_github.sh
./setup_github.sh
```

---

**Good luck with your IndiaAI Challenge submission! 🏆**


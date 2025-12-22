#!/bin/bash

# Quick GitHub Upload Script
# Run this after creating your GitHub repository

echo "╔══════════════════════════════════════════════════════╗"
echo "║  DocSynthesis-V1 GitHub Upload Script               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install Git first.${NC}"
    exit 1
fi

echo -e "${BLUE}📝 Before running this script, make sure you have:${NC}"
echo "   1. Created a GitHub repository (https://github.com/new)"
echo "   2. Named it 'docsynthesis-v1' (or similar)"
echo "   3. NOT initialized it with README, .gitignore, or license"
echo ""

read -p "Have you created the GitHub repository? [y/N]: " created
if [[ ! $created =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Please create a GitHub repository first, then run this script again.${NC}"
    echo ""
    echo "Instructions:"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: docsynthesis-v1"
    echo "3. Description: Intelligent Document Processing for IndiaAI Challenge"
    echo "4. Choose Public or Private"
    echo "5. Do NOT initialize with any files"
    echo "6. Click 'Create repository'"
    exit 0
fi

# Get GitHub username
echo ""
read -p "Enter your GitHub username: " username
if [ -z "$username" ]; then
    echo -e "${RED}❌ Username cannot be empty${NC}"
    exit 1
fi

# Get repository name
read -p "Enter repository name (default: docsynthesis-v1): " reponame
reponame=${reponame:-docsynthesis-v1}

# Confirm
echo ""
echo -e "${BLUE}Repository URL will be:${NC}"
echo "https://github.com/$username/$reponame"
echo ""
read -p "Is this correct? [Y/n]: " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo ""
    echo "🔧 Initializing Git repository..."
    git init
    echo -e "${GREEN}✓${NC} Git initialized"
else
    echo ""
    echo -e "${YELLOW}⚠${NC} Git already initialized"
fi

# Add all files
echo ""
echo "📦 Adding files..."
git add .
echo -e "${GREEN}✓${NC} Files added"

# Create initial commit
echo ""
echo "💾 Creating initial commit..."
git commit -m "Initial commit: DocSynthesis-V1 - IndiaAI IDP Challenge submission

Complete implementation including:
- DeepSeek-OCR with Context Optical Compression
- HybriDLA layout analysis
- Many-to-one multilingual NMT
- Explainable AI with Feature Alignment Metrics
- REST API server
- Docker deployment
- Comprehensive documentation

Technical submission for IndiaAI Intelligent Document Processing Challenge."

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Commit created"
else
    echo -e "${RED}❌ Commit failed${NC}"
    exit 1
fi

# Set main branch
echo ""
echo "🌿 Setting main branch..."
git branch -M main
echo -e "${GREEN}✓${NC} Branch set to main"

# Add remote
echo ""
echo "🔗 Adding remote repository..."
git remote add origin "https://github.com/$username/$reponame.git"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Remote added"
else
    echo -e "${YELLOW}⚠${NC} Remote already exists, updating..."
    git remote set-url origin "https://github.com/$username/$reponame.git"
fi

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
echo -e "${YELLOW}You may be prompted for GitHub credentials:${NC}"
echo "   Username: Your GitHub username"
echo "   Password: Use a Personal Access Token (not your password!)"
echo "   Generate token at: https://github.com/settings/tokens"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  🎉 Successfully uploaded to GitHub! 🎉              ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}✓${NC} Repository URL:"
    echo "   https://github.com/$username/$reponame"
    echo ""
    echo "📋 Next Steps:"
    echo ""
    echo "1. Visit your repository:"
    echo "   https://github.com/$username/$reponame"
    echo ""
    echo "2. Add topics (optional but recommended):"
    echo "   - Click gear icon next to 'About'"
    echo "   - Add: ocr, document-processing, ai, indiaai, deepseek"
    echo ""
    echo "3. Create a release:"
    echo "   git tag -a v1.0.0 -m 'IndiaAI Challenge Submission v1.0.0'"
    echo "   git push origin v1.0.0"
    echo "   Then go to Releases on GitHub and create from tag"
    echo ""
    echo "4. Update README.md:"
    echo "   - Replace 'yourusername' with '$username'"
    echo "   - Add any custom badges or links"
    echo ""
    echo "5. Submit to competition:"
    echo "   - Use URL: https://github.com/$username/$reponame"
    echo "   - Include in submission form"
    echo ""
    echo "🏆 Good luck with your IndiaAI submission!"
else
    echo ""
    echo -e "${RED}❌ Push failed${NC}"
    echo ""
    echo "Common issues:"
    echo "1. Authentication failed:"
    echo "   - Use Personal Access Token instead of password"
    echo "   - Generate at: https://github.com/settings/tokens"
    echo ""
    echo "2. Repository doesn't exist:"
    echo "   - Make sure you created the repository on GitHub first"
    echo "   - Check the repository name is correct"
    echo ""
    echo "3. Permission denied:"
    echo "   - Make sure you have write access to the repository"
    echo ""
    echo "Try again with: git push -u origin main"
    exit 1
fi


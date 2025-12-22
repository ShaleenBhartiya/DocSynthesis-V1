#!/usr/bin/env python3
"""
Test Gradio Demo Components
Quick verification that all dependencies are available
"""

import sys

print("🧪 Testing DocSynthesis-V1 Gradio Demo Components...")
print("=" * 60)

# Test imports
tests_passed = 0
tests_failed = 0

def test_import(module_name, display_name=None):
    """Test if a module can be imported."""
    global tests_passed, tests_failed
    display = display_name or module_name
    try:
        __import__(module_name)
        print(f"✅ {display}: OK")
        tests_passed += 1
        return True
    except ImportError as e:
        print(f"❌ {display}: FAILED - {e}")
        tests_failed += 1
        return False

# Core dependencies
print("\n📦 Core Dependencies:")
test_import("gradio", "Gradio")
test_import("PIL", "Pillow (PIL)")
test_import("numpy", "NumPy")
test_import("cv2", "OpenCV")
test_import("plotly", "Plotly")

# Optional dependencies
print("\n📦 Optional Dependencies:")
test_import("torch", "PyTorch")
test_import("transformers", "Transformers")

# Check Python version
print("\n🐍 Python Version:")
version_info = sys.version_info
if version_info >= (3, 9):
    print(f"✅ Python {sys.version.split()[0]}: OK")
    tests_passed += 1
else:
    print(f"⚠️  Python {sys.version.split()[0]}: Recommended 3.9+")
    tests_failed += 1

# Test file existence
print("\n📁 Required Files:")
import os

files_to_check = [
    "gradio_app.py",
    "requirements-gradio.txt",
    "launch_demo.sh",
    "GRADIO_QUICKSTART.md",
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file}: Found")
        tests_passed += 1
    else:
        print(f"❌ {file}: Missing")
        tests_failed += 1

# Summary
print("\n" + "=" * 60)
print(f"📊 Test Results: {tests_passed} passed, {tests_failed} failed")

if tests_failed == 0:
    print("\n🎉 All tests passed! Ready to launch the demo.")
    print("\n🚀 Run: ./launch_demo.sh")
    sys.exit(0)
else:
    print("\n⚠️  Some tests failed. Please install missing dependencies:")
    print("   pip install -r requirements-gradio.txt")
    sys.exit(1)


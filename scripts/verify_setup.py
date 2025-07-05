#!/usr/bin/env python3
"""
Setup verification script for Colombia RAG Chatbot.
"""

import sys
import os
import importlib
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version >= (3, 8):
        return True, f" Python {version.major}.{version.minor}.{version.micro}"
    return False, f" Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"


def check_project_structure() -> Tuple[bool, str]:
    """Check if project structure is correct."""
    required_dirs = [
        "app",
        "app/config",
        "app/core", 
        "app/models",
        "app/services",
        "app/api",
        "app/utils",
        "data/raw",
        "data/processed", 
        "data/vectorstore",
        "tests",
        "scripts",
        "requirements"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    if not missing_dirs:
        return True, " Project structure is correct"
    return False, f" Missing directories: {', '.join(missing_dirs)}"


def check_required_files() -> Tuple[bool, str]:
    """Check if required files exist."""
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/config/settings.py",
        "app/config/logging.py",
        "app/core/exceptions.py",
        "pyproject.toml",
        "requirements/base.txt",
        "requirements/dev.txt",
        "requirements/prod.txt",
        ".env.example",
        ".gitignore",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        return True, " All required files present"
    return False, f" Missing files: {', '.join(missing_files)}"


def check_imports() -> List[Tuple[bool, str]]:
    """Check if core modules can be imported."""
    modules_to_check = [
        ("app.config.settings", "Settings configuration"),
        ("app.config.logging", "Logging configuration"),
        ("app.core.exceptions", "Custom exceptions"),
        ("app.main", "Main FastAPI application"),
    ]
    
    results = []
    
    for module_name, description in modules_to_check:
        try:
            importlib.import_module(module_name)
            results.append((True, f"{description} imports successfully"))
        except ImportError as e:
            results.append((False, f"{description} import failed: {e}"))
        except Exception as e:
            results.append((False, f"{description} error: {e}"))
    
    return results


def check_dependencies() -> List[Tuple[bool, str]]:
    """Check if key dependencies can be imported."""
    dependencies = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("structlog", "Structured logging"),
        ("requests", "HTTP client"),
        ("bs4", "HTML parsing"),
        ("langchain", "LangChain framework"),
        ("sentence_transformers", "Sentence transformers"),
        ("chromadb", "Vector database"),
    ]
    
    results = []
    
    for module_name, description in dependencies:
        try:
            importlib.import_module(module_name)
            results.append((True, f"{description} is available"))
        except ImportError:
            results.append((False, f"{description} not installed"))
    
    return results


def check_environment() -> Tuple[bool, str]:
    """Check if environment can be loaded."""
    try:
        from app.config.settings import settings
        return True, f"Environment loaded (mode: {settings.environment})"
    except Exception as e:
        return False, f"Environment loading failed: {e}"


def main():
    """Run all verification checks."""
    print("🔍 Verifying Colombia RAG Chatbot setup...\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Required Files", check_required_files),
        ("Environment", check_environment),
    ]
    
    all_passed = True
    
    # Run basic checks
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            print(f"{check_name}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"{check_name}: Error running check: {e}")
            all_passed = False
    
    print()
    
    # Check imports
    print("Checking module imports:")
    import_results = check_imports()
    for passed, message in import_results:
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    print()
    
    # Check dependencies
    print("Checking dependencies:")
    dependency_results = check_dependencies()
    for passed, message in dependency_results:
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    print()
    
    # Final result
    if all_passed:
        print("All checks passed! Your setup is ready.")
        print("You can now start the development server with:")
        print("  python -m app.main")
        return 0
    else:
        print("Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
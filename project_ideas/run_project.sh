#!/bin/bash

# Quick runner script for OPM project ideas
# Usage: ./run_project.sh [project_number]
# Example: ./run_project.sh 1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    PYTHON="python"
else
    # Use the venv python directly
    PYTHON="$REPO_ROOT/.venv/bin/python"
fi

# Project mapping
declare -A PROJECTS
PROJECTS[1]="01_medical_diagnosis/medical_diagnosis_opm.py"
PROJECTS[2]="02_spam_detection/spam_detection_opm.py"
PROJECTS[3]="03_customer_churn/customer_churn_opm.py"
PROJECTS[4]="04_sentiment_analysis/sentiment_analysis_opm.py"

# Display menu if no argument
if [ -z "$1" ]; then
    echo "========================================"
    echo "OPM Project Ideas - Quick Runner"
    echo "========================================"
    echo ""
    echo "Available projects:"
    echo "  1. Medical Diagnosis OPM"
    echo "  2. Spam Detection OPM"
    echo "  3. Customer Churn Prediction OPM"
    echo "  4. Sentiment Analysis OPM"
    echo ""
    echo "Usage: $0 [project_number]"
    echo "Example: $0 1"
    echo ""
    echo "Or run directly:"
    echo "  $PYTHON 01_medical_diagnosis/medical_diagnosis_opm.py"
    exit 0
fi

# Run selected project
if [ -n "${PROJECTS[$1]}" ]; then
    PROJECT_PATH="$SCRIPT_DIR/${PROJECTS[$1]}"
    if [ -f "$PROJECT_PATH" ]; then
        echo "Running project $1..."
        echo ""
        $PYTHON "$PROJECT_PATH"
    else
        echo "Error: Project file not found: $PROJECT_PATH"
        exit 1
    fi
else
    echo "Error: Invalid project number: $1"
    echo "Please choose 1, 2, 3, or 4"
    exit 1
fi

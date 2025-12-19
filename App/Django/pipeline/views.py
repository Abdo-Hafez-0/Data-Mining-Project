from django.shortcuts import render
from django.conf import settings
from pathlib import Path

from scripts.cleaning import run_cleaning
from scripts.eda import run_eda
from scripts.features import run_feature_engineering
from scripts.modeling import train_model

# Django base directory (…/App/Django)
BASE_DIR = Path(settings.BASE_DIR)

# Project root directory (…/Project)
PROJECT_ROOT = BASE_DIR.parent.parent

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw"
CLEANED_DATA_DIR = PROJECT_ROOT / "Data" / "Cleaned"
FEATURED_DATA_DIR = PROJECT_ROOT / "Data" / "Featured"

RAW_DATA_FILE = RAW_DATA_DIR / "Raw_Data.csv"
CLEANED_DATA_FILE = CLEANED_DATA_DIR / "cleaned_data.csv"
FEATURED_DATA_FILE = FEATURED_DATA_DIR / "featured_data.csv"

def home(request):
    return render(request, 'pipeline/home.html')

def cleaning_page(request):
    summary = run_cleaning(
        RAW_DATA_FILE,
        CLEANED_DATA_FILE
    )
    return render(request, 'pipeline/cleaning.html', summary)


def eda_page(request):
    stats = run_eda(CLEANED_DATA_FILE)
    return render(request, 'pipeline/eda.html', stats)

def features_page(request):
    info = run_feature_engineering(
        CLEANED_DATA_FILE,
        FEATURED_DATA_FILE
    )
    return render(request, 'pipeline/features.html', info)

def modeling_page(request):
    results = train_model(FEATURED_DATA_FILE)
    return render(request, 'pipeline/modeling.html', results)

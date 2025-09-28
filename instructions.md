Tasks:
1. Go through the files README.md and Architecture.md to understand the project requirements and architecture.
2. Generate te code for all the other files accordingly.

Here is some information related to the project:
📁 Complete Project Structure
I've created all the necessary files and boilerplate templates:
Core Application Files

app.py - Main Streamlit application with upload, dashboard, and interactive Q&A
config.py - Configuration settings and API management
requirements.txt - All necessary Python dependencies
setup.py - Automated setup script

Data Processing Layer

utils/data_utils.py - Data cleaning, validation, and metrics computation
utils/chart_utils.py - Chart generation functions (funnel, risk analysis, site distribution)
utils/insights_engine.py - Natural language query processing

AI Services

services/ai_summary.py - AI-powered summary generation for different audiences

Data Generation

data/generate_data.py - Synthetic data generator (200 participants with realistic patterns)

Styling & Documentation

assets/styles.css - Custom CSS for professional appearance
README.md - Comprehensive documentation
ARCHITECTURE.md - Detailed technical architecture guide

🚀 Key Features Implemented
1. Data Upload & Processing

CSV upload with automatic validation
Data cleaning and standardization
Missing value handling
Derived metrics calculation

2. Interactive Dashboard

Key KPI metrics cards
Pre-built visualizations (eligibility funnel, dropout risk, site performance)
Responsive design with professional styling

3. AI-Powered Summaries

Executive summary (strategic focus)
Clinical insights (safety & compliance)
Marketing report (recruitment optimization)
Fallback templates when AI isn't available

4. Natural Language Queries ⭐ (Your Requested Feature)

Ask questions in plain English
Pattern matching for common queries
AI-powered complex query analysis
Dynamic chart generation
Contextual explanations

🎯 What This Application Does
Core Functionality

Upload CSV data → Automatic cleaning and validation
Generate dashboard → KPIs, charts, and AI summaries
Interactive Q&A → Natural language queries with dynamic responses
Multi-audience reports → Executive, clinical, and marketing views

Sample Queries It Can Handle

"Show dropout risk by age group"
"Which site has the highest eligibility rate?"
"Compare sites by performance"
"How many high-risk participants are there?"
"Create a chart of participants by location"

🏗 Architecture Summary
4-Layer Architecture

UI Layer: Streamlit frontend with upload, dashboard, and Q&A sections
Logic Layer: Data processing, query parsing, session management
Service Layer: AI integration (OpenAI), chart generation
Data Layer: CSV files, session state, synthetic data generation

AI Integration

Supports OpenAI API for AI-powered features
Intelligent prompt engineering for different audiences
Natural language query understanding
Fallback to template-based responses

Data Flow
CSV Upload → Data Cleaning → Metrics Computation → Dashboard Generation → AI Summaries → Interactive Q&A
⚡ Quick Start

Install dependencies: pip install -r requirements.txt
Generate sample data: python data/generate_data.py
Set up API keys: Copy .env.template to .env and add your keys (optional)
Run application: streamlit run app.py
Open browser: http://localhost:8501

🎬 Demo Flow (20 minutes)

Upload/Load Data (3 min) → Show data processing
Dashboard Tour (7 min) → KPIs, charts, AI summaries
Interactive Queries (7 min) → Natural language questions
Advanced Features (3 min) → Custom charts, comparisons

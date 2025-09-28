# Clinical Trial Analytics Demo

A comprehensive Streamlit application for analyzing clinical trial participant data with AI-powered insights and interactive visualizations.

## 🚀 Features

### Core Functionality
- **Data Upload & Processing**: Upload CSV files with participant data
- **Interactive Dashboard**: Key metrics, charts, and visualizations
- **AI-Powered Summaries**: Executive, clinical, and marketing reports
- **Natural Language Queries**: Ask questions about your data in plain English
- **Custom Chart Generation**: Create visualizations based on user queries

### Key Metrics Tracked
- Participant eligibility rates
- Dropout risk analysis
- Site performance comparison
- Age group analysis
- Protocol deviation tracking

### AI Capabilities
- Automated report generation for different audiences
- Natural language query processing
- Custom insight generation
- Pattern recognition and recommendations

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd demo_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env file with your API keys
   ```

4. **Generate sample data**
   ```bash
   python data/generate_data.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📊 Data Format

### Required Columns
- `participant_id`: Unique identifier
- `age`: Participant age (numeric)
- `location`: Site/location name
- `meets_criteria`: Boolean (True/False, 1/0, Yes/No)
- `dropout_risk`: Risk level (High/Medium/Low or 1/2/3)

### Optional Columns
- `gender`: Participant gender
- `enrollment_date`: Date of enrollment
- `ethnicity`: Participant ethnicity
- `bmi`: Body mass index
- `protocol_deviation`: Any protocol deviations
- `followup_status`: Current status (Active/Withdrawn/Lost)

### Sample Data Structure
```csv
participant_id,age,gender,location,meets_criteria,dropout_risk,enrollment_date
P0001,45,Female,Site A,True,Low,2024-01-15
P0002,62,Male,Site B,True,Medium,2024-01-16
P0003,71,Female,Site A,False,High,2024-01-17
```

## � Quick Start

### Automated Setup
```bash
python setup.py
```

### Manual Setup
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Generate sample data**: `python data/generate_data.py`
3. **Run application**: `streamlit run app.py`

## �🔧 Configuration

### API Setup

The application supports both Azure OpenAI and OpenAI APIs:

#### Azure OpenAI (Recommended)
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
```

#### OpenAI (Alternative)
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### Environment Setup
1. Copy `.env.template` to `.env`
2. Add your API credentials
3. Restart the application

## 🎯 Features Overview

### 📁 Data Upload & Management
- **CSV File Upload**: Drag-and-drop or browse to upload participant data
- **Data Validation**: Automatic validation of required columns and data types
- **Sample Data**: Pre-generated synthetic data for immediate testing
- **Data Quality Checks**: Missing value detection and data completeness analysis

### 📊 Interactive Dashboard
- **Key Performance Indicators**: Participant counts, eligibility rates, risk metrics
- **Eligibility Funnel**: Visual representation of participant screening flow
- **Risk Distribution**: Breakdown of dropout risk across participants
- **Site Performance**: Comparative analysis across study locations
- **Demographics Analysis**: Age distribution, gender breakdown, BMI categories

### 🤖 AI-Powered Insights
- **Multi-Audience Summaries**: Executive, clinical, and marketing reports
- **Natural Language Queries**: Ask questions in plain English
- **Intelligent Analysis**: Pattern recognition and trend identification
- **Custom Insights**: Tailored recommendations based on your data

### 📈 Custom Chart Builder
- **Multiple Chart Types**: Bar, pie, histogram, scatter, box, line, correlation heatmap
- **Interactive Filtering**: Filter by age, site, risk level, eligibility status
- **Dynamic Grouping**: Group data by categorical variables
- **Preset Gallery**: Quick access to common visualizations

### ⚙️ Advanced Features
- **Session Management**: Persistent data across page navigation
- **Real-time Processing**: Instant updates as you interact with data
- **Error Handling**: Graceful handling of data issues and API failures
- **Template Fallbacks**: Functional without AI API configuration

## 📊 Supported Data Formats

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | String | Unique participant identifier |
| `age` | Numeric | Participant age in years |
| `location` | String | Study site or location |
| `meets_criteria` | Boolean | Eligibility status (True/False, Yes/No, 1/0) |
| `dropout_risk` | String | Risk level (High/Medium/Low, 1/2/3) |

### Optional Columns
| Column | Type | Description |
|--------|------|-------------|
| `gender` | String | Participant gender |
| `enrollment_date` | Date | Date of enrollment (YYYY-MM-DD) |
| `ethnicity` | String | Participant ethnicity |
| `bmi` | Numeric | Body mass index |
| `protocol_deviation` | Boolean | Protocol deviation flag |
| `followup_status` | String | Current status (Active/Withdrawn/Lost/Completed) |

## 🎨 User Interface Guide

### Navigation
- **📁 Data Upload**: Upload and manage your clinical trial data
- **📊 Dashboard**: View comprehensive analytics and key metrics
- **💬 Interactive Q&A**: Ask questions and get AI-powered insights
- **📈 Custom Charts**: Create custom visualizations
- **⚙️ Settings**: Configure application and view data information

### Sidebar Features
- **Data Status**: Current data loading status
- **AI Connection**: Test and monitor AI service connectivity
- **Reset Application**: Clear all data and start fresh

## 🔧 Configuration

#### OpenAI API
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

**Note**: If no API keys are provided, the application will use template-based summaries.

## 📖 Usage Guide

### 1. Upload Data
- Click "Choose a CSV file" or use "Load Sample Data"
- Preview your data to ensure correct formatting
- The system automatically cleans and validates data

### 2. View Dashboard
- Key metrics displayed as cards
- Pre-built visualizations (funnel, risk distribution, site comparison)
- AI-generated summaries for different audiences

### 3. Interactive Queries
Ask natural language questions like:
- "Show dropout risk by age group"
- "Which site has the highest eligibility rate?"
- "Compare sites by performance"
- "How many high-risk participants do we have?"

### 4. AI Summaries
Choose from three tailored reports:
- **Executive**: High-level metrics and recommendations
- **Clinical**: Patient safety and protocol insights
- **Marketing**: Recruitment and site optimization

## 🏗 Architecture

### High-Level Components

```
┌─────────────────────────────────────┐
│          Streamlit Frontend         │
│  Upload │ Dashboard │ AI Insights   │
└─────────────────────────────────────┘
           │                    │
           ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  Data Processing │  │   AI Services    │
│  (Pandas)        │  │  (OpenAI/Azure)  │
└──────────────────┘  └──────────────────┘
           │                    │
           ▼                    ▼
┌─────────────────────────────────────┐
│     Visualization & Insights        │
│   (Plotly Charts & Reports)         │
└─────────────────────────────────────┘
```

### File Structure
```
demo_app/
├── app.py                 # Main application
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
├── .env.template          # Environment template
├── README.md              # Documentation
│
├── data/
│   ├── generate_data.py   # Synthetic data generator
│   └── participants_sample.csv  # Sample dataset
│
├── utils/
│   ├── data_utils.py      # Data processing
│   ├── chart_utils.py     # Chart generation
│   └── insights_engine.py # Query processing
│
├── services/
│   └── ai_summary.py      # AI integration
│
└── assets/
    └── styles.css         # Custom styling
```

## 🎯 Key Features Explained

### 1. Data Processing Pipeline
- **Automatic Cleaning**: Handles missing values, data type conversion
- **Validation**: Ensures required columns exist
- **Derived Metrics**: Generates age groups, risk scores
- **Site Analysis**: Computes per-site performance metrics

### 2. AI-Powered Analytics
- **Smart Summaries**: Context-aware reports for different stakeholders
- **Query Understanding**: Natural language processing for user questions
- **Pattern Recognition**: Identifies trends and anomalies
- **Recommendation Engine**: Suggests actionable insights

### 3. Interactive Visualizations
- **Pre-built Charts**: Eligibility funnel, risk distribution, site performance
- **Dynamic Generation**: Charts created based on user queries
- **Responsive Design**: Works on desktop and mobile devices
- **Export Capabilities**: Save charts and data

### 4. Multi-Audience Reports
- **Executive Dashboard**: KPIs, trends, strategic recommendations
- **Clinical Insights**: Safety signals, protocol compliance, patient characteristics
- **Marketing Analytics**: Recruitment metrics, site optimization, ROI analysis

## 🔍 Query Examples

### Basic Questions
- "How many participants do we have?"
- "What's the eligibility rate?"
- "Show me the age distribution"

### Comparative Analysis
- "Compare sites by eligibility rate"
- "Which age group has highest dropout risk?"
- "Show performance by location"

### Risk Analysis
- "Who are the high-risk participants?"
- "Dropout risk by site"
- "Risk factors by age group"

### Custom Visualizations
- "Create a bar chart of participants by site"
- "Show correlation between age and risk"
- "Plot eligibility over time"

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Connection Issues**
   - Check your `.env` file configuration
   - Verify API keys are correct
   - Test with template mode first

3. **Data Upload Problems**
   - Ensure CSV has required columns
   - Check for special characters in data
   - Verify file is not corrupted

4. **Performance Issues**
   - Large datasets (>1000 rows) may be slower
   - Consider sampling for demo purposes
   - Check available system memory

### Debug Mode
Enable debug mode in `.env`:
```env
DEBUG_MODE=True
```

## 📈 Demo Flow

### Recommended Presentation Sequence

1. **Introduction** (2 mins)
   - Show upload interface
   - Load sample data
   - Explain data structure

2. **Dashboard Overview** (5 mins)
   - Walk through key metrics
   - Explain each visualization
   - Highlight insights

3. **AI Summaries** (3 mins)
   - Show executive summary
   - Switch to clinical view
   - Demonstrate marketing report

4. **Interactive Queries** (5 mins)
   - Ask pre-planned questions
   - Show chart generation
   - Demonstrate flexibility

5. **Advanced Features** (3 mins)
   - Custom filtering
   - Data export options
   - Performance metrics

6. **Q&A and Extensions** (2 mins)
   - Answer questions
   - Discuss customization
   - Next steps

## 🔮 Future Enhancements

### Phase 2 Features
- **Real-time Data Integration**: Connect to live databases
- **Advanced ML Models**: Predictive analytics for dropout risk
- **Workflow Integration**: Ticket creation, alerts, notifications
- **Multi-study Support**: Compare across different trials
- **Advanced Security**: User authentication, role-based access

### Technical Improvements
- **Caching Layer**: Improve performance for large datasets
- **Background Processing**: Async data processing
- **Database Integration**: PostgreSQL, MongoDB support
- **API Development**: REST API for external integrations

## 📞 Support

For technical issues or questions:
1. Check this documentation first
2. Review error logs in the console
3. Test with sample data
4. Verify API configurations

## 🏷 Version History

- **v1.0**: Initial demo version with core features
- **v1.1**: Added natural language queries
- **v1.2**: Enhanced AI summaries
- **v1.3**: Improved visualizations and styling

---

**Built with**: Streamlit, Pandas, Plotly, OpenAI API
**Demo Duration**: ~20 minutes
**Target Audience**: Clinical research teams, executives, data analysts
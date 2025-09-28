"""
Clinical Trial Analytics Demo - Setup Script
Automated setup for the clinical trial analytics application
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version check passed: {sys.version}")
        return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "assets", 
        "pages",
        "services",
        "utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created/verified: {directory}/")
    
    return True

def setup_environment():
    """Setup environment configuration"""
    print("\n⚙️ Setting up environment configuration...")
    
    env_template = ".env.template"
    env_file = ".env"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_template):
            shutil.copy(env_template, env_file)
            print(f"  ✅ Created {env_file} from template")
            print(f"  📝 Please edit {env_file} with your API keys")
        else:
            print(f"  ⚠️ No environment template found")
    else:
        print(f"  ✅ Environment file {env_file} already exists")
    
    return True

def generate_sample_data():
    """Generate sample data if it doesn't exist"""
    print("\n📊 Generating sample data...")
    
    data_file = "data/participants_sample.csv"
    
    if not os.path.exists(data_file):
        try:
            subprocess.check_call([sys.executable, "data/generate_data.py"])
            print("✅ Sample data generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate sample data: {e}")
            return False
    else:
        print("✅ Sample data already exists")
    
    return True

def verify_installation():
    """Verify that the installation is working"""
    print("\n🔍 Verifying installation...")
    
    # Check if all required files exist
    required_files = [
        "app.py",
        "config.py",
        "requirements.txt",
        "utils/data_utils.py",
        "utils/chart_utils.py",
        "utils/insights_engine.py",
        "services/ai_summary.py",
        "data/generate_data.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    # Try importing key modules
    try:
        import streamlit
        import pandas
        import plotly
        import openai
        print("  ✅ All required packages can be imported")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    
    print("\n🎉 Installation verification complete!")
    return True

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. 🔑 Configure API Keys:")
    print("   • Edit .env file with your OpenAI or Azure OpenAI credentials")
    print("   • See .env.template for required variables")
    
    print("\n2. 🏃‍♂️ Run the Application:")
    print("   streamlit run app.py")
    
    print("\n3. 📊 Using the App:")
    print("   • Upload your CSV data OR use the sample data")
    print("   • Navigate through the dashboard pages")
    print("   • Try the AI-powered insights (requires API keys)")
    
    print("\n4. 📁 Sample Data:")
    print("   • Sample data is available in data/participants_sample.csv")
    print("   • Use 'Load Sample Data' button to try the app immediately")
    
    print("\n💡 Tips:")
    print("   • Required CSV columns: participant_id, age, location, meets_criteria, dropout_risk")
    print("   • Optional columns: gender, enrollment_date, ethnicity, bmi, protocol_deviation")
    print("   • AI features work without API keys using template responses")
    
    print("\n🆘 Need Help?")
    print("   • Check README.md for detailed documentation")
    print("   • Verify all dependencies are installed correctly")
    print("   • Ensure your CSV data follows the required format")

def main():
    """Main setup function"""
    print("🏥 Clinical Trial Analytics Demo - Setup")
    print("="*50)
    
    # Run setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Setting up environment", setup_environment),
        ("Generating sample data", generate_sample_data),
        ("Verifying installation", verify_installation)
    ]
    
    for step_name, step_function in steps:
        print(f"\n🔄 {step_name}...")
        
        if not step_function():
            print(f"\n❌ Setup failed at step: {step_name}")
            print("Please fix the errors above and run setup again.")
            sys.exit(1)
    
    # Print success message and next steps
    print_next_steps()

if __name__ == "__main__":
    main()
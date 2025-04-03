# Chronic Disease Management System

## Demo Video
[[YouTube]](https://youtu.be/K-ivPbrrKIM)

## Mobile App Link
[[Cardio Mobile App]](https://github.com/sjamillah/cardio_mobile_app.git)

## Project Description

### Background
Chronic diseases are a critical health challenge in sub-Saharan Africa, accounting for 37% of deaths and increasing from 24% in 2000. This project addresses the significant healthcare gaps in resource-limited settings by leveraging advanced machine learning technologies to provide personalized health risk assessments.

### Problem Statement
In sub-Saharan Africa, healthcare systems face multiple challenges:
- Limited medical infrastructure
- Lack of early disease detection mechanisms
- Insufficient personalized healthcare interventions
- High mortality rates from chronic diseases

### Solution Overview
The Chronic Disease Management System is a comprehensive mobile application that:
- Utilizes machine learning algorithms to predict chronic disease risks
- Provides personalized health recommendations
- Supports early detection and intervention
- Functions effectively in low-resource healthcare environments

### Key Features
1. Advanced Machine Learning Risk Prediction
   - Predicts risks for cardiovascular diseases, diabetes, and other chronic conditions
   - 85% accuracy across multiple disease categories
   - Uses 14 comprehensive health indicators

2. Personalized Health Insights
   - Generates individualized risk assessments
   - Offers actionable health recommendations
   - Supports continuous health monitoring

3. Cultural Adaptability
   - Designed specifically for sub-Saharan African healthcare contexts
   - Supports multiple languages and cultural considerations
   - Works with low-bandwidth and offline environments

4. Comprehensive Risk Assessment
   - Analyzes multiple health parameters
   - Considers demographic and lifestyle factors
   - Provides nuanced risk categorization

## Prerequisites

### Software Requirements
- Flutter SDK (Version 2.10 or higher)
- Dart SDK
- Python 3.8+
- pip
- Virtual environment support

### Hardware Requirements
- Minimum 4GB RAM
- 10GB free disk space
- Stable internet connection for initial setup

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/sjamillah/Cardio-Vascular_Pipeline.git
cd Cardio-Vascular_Pipeline
```

### 2. Mobile App Setup
```bash
# Navigate to mobile app directory
cd cardio_mobile_app

# Get Flutter dependencies
flutter pub get

# Verify installation
flutter doctor

# Run the app
flutter run
```

### 3. Backend API Setup
```bash
# Navigate to backend directory
cd ../src

# Create virtual environment
python -m venv venv
source env/bin/activate  # On Windows: env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Run API server
uvicorn main:app --reload
```

### Configuration

#### Mobile App Configuration
Create or modify `.env` in the mobile app directory:
```
API_BASE_URL=[https://your-backend-api.com](https://cardio-vascular-pipeline.onrender.com)
LOGGING_ENABLED=true
OFFLINE_MODE=true
```

#### Backend Configuration
Create or modify `.env` in the backend directory:
```
DATABASE_URL=postgresql://username:password@localhost/cdms
MODEL_PATH=./models/
DEBUG=false
ALLOWED_HOSTS=localhost,127.0.0.1
```

## How to Use the Application

1. Install the mobile application
2. Navigate to the Risk Prediction Screen
3. Enter Health Parameters
   - Input key health indicators (age, height, weight, blood pressure, etc.)
   - Provide additional health-related information
4. Generate Risk Assessment
   - Receive immediate personalized risk prediction
   - View detailed risk probability
5. Explore Feature Visualizations
   - Access graphical representations of health features
   - Understand how different factors contribute to risk
6. Optional: Bulk Data Retraining
   - Upload CSV file with multiple health records
   - Contribute to model improvement
   - Retrain the model using selected features

## Troubleshooting

### Common Installation Issues
- Ensure all prerequisites are installed
- Check Flutter and Dart version compatibility
- Verify internet connection
- Confirm Python virtual environment activation

### Reporting Issues
- Check GitHub Issues section
- Provide detailed error logs
- Include your environment details

## Technology Stack

### Frontend
- Flutter
- Dart
- Provider for state management

### Backend
- Python
- FastAPI
- TensorFlow
- Scikit-learn

### Database
- MongoDB

## Performance Metrics
- Prediction Accuracy: 85%
- Supported Platforms: iOS, Android
- Supported Languages: English, French (Initial Release)

## License
MIT License

## Contact
**Jamillah SSOZI**
- Email: j.ssozi@alustudent.com
- Institution: African Leadership University

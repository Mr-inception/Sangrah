# Credit Risk Model Web Application

A modern, interactive web application for credit risk prediction using AI and machine learning.

## 🌟 Features

- **Interactive Web Interface**: Beautiful, responsive design with modern UI/UX
- **Real-time Predictions**: Live credit risk assessment using trained AI models
- **Multiple Sections**: Overview, Features, Live Demo, Code Examples, and About
- **Error Handling**: Robust error handling and user feedback
- **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites

1. Python 3.8 or higher
2. All dependencies from `requirements.txt`

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Web Application**:
   ```bash
   python app.py
   ```

3. **Access the Webpage**:
   Open your browser and go to: `http://localhost:5000`

## 📁 Project Structure

```
├── app.py                          # Flask web application
├── credit_risk_model.py            # Main ML model class
├── data.py                         # Data generation script
├── simulated_credit_risk_data.csv  # Sample dataset
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates
│   └── index.html                 # Main webpage
├── README.md                       # Project documentation
├── WEB_README.md                   # Web app documentation
└── test_app.py                     # Test script
```

## 🎯 Web Application Features

### 1. Overview Section
- Project introduction and key benefits
- Clear explanation of credit risk modeling
- Visual presentation of advantages

### 2. Features Section
- Interactive feature cards with icons
- Detailed descriptions of each capability
- Hover effects and animations

### 3. Live Demo Section
- **Interactive Form**: User-friendly input form for credit risk assessment
- **Real-time Predictions**: Instant results using trained AI models
- **Comprehensive Results**: Risk level, default probability, and recommendations
- **Error Handling**: Graceful error messages and loading states

### 4. Code Examples Section
- Syntax-highlighted code blocks
- Practical usage examples
- Copy-paste ready code snippets

### 5. About Section
- Technology stack information
- Model performance metrics
- Project structure overview

## 🔧 Technical Details

### Backend (Flask)
- **Framework**: Flask 2.0+
- **Model Integration**: Seamless integration with trained ML models
- **API Endpoints**: RESTful API for predictions
- **Error Handling**: Comprehensive error handling and validation

### Frontend (HTML/CSS/JavaScript)
- **Design**: Modern, responsive design with gradient backgrounds
- **Framework**: Vanilla JavaScript with modern ES6+ features
- **Styling**: CSS3 with animations and transitions
- **Icons**: Font Awesome icons for enhanced UX

### Key Technologies
- **Python**: Core programming language
- **Flask**: Web framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **HTML5/CSS3**: Frontend markup and styling
- **JavaScript**: Interactive functionality

## 🎨 UI/UX Features

### Design Elements
- **Gradient Backgrounds**: Modern purple-blue gradients
- **Card-based Layout**: Clean, organized information presentation
- **Smooth Animations**: Hover effects and transitions
- **Responsive Design**: Mobile-first approach
- **Loading States**: Spinner animations for better UX

### Interactive Elements
- **Navigation**: Tab-based navigation with active states
- **Forms**: Validated input forms with real-time feedback
- **Results Display**: Clean, organized result presentation
- **Error Handling**: User-friendly error messages

## 🔍 API Endpoints

### POST /predict
Predicts credit risk for given customer data.

**Request Body**:
```json
{
    "age": 35,
    "monthly_income": 50000,
    "on_time_utility_payments": 90,
    "job_stability": 5,
    "social_score": 0.7,
    "ecommerce_monthly_spend": 8000,
    "phone_usage_score": 0.8
}
```

**Response**:
```json
{
    "success": true,
    "risk_level": "Low",
    "default_probability": "5.2%",
    "recommendation": "Approve application",
    "prediction": 0,
    "probability": 0.052
}
```

### GET /api/model-info
Returns information about the loaded model.

**Response**:
```json
{
    "model_type": "Random Forest",
    "features": ["age", "monthly_income", "on_time_utility_payments", "job_stability", "social_score", "ecommerce_monthly_spend", "phone_usage_score"],
    "status": "loaded"
}
```

## 🚀 Deployment

### Local Development
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access at: `http://localhost:5000`

### Production Deployment
For production deployment, consider:
- Using a production WSGI server (e.g., Gunicorn)
- Setting up environment variables
- Implementing proper logging
- Adding security headers
- Using HTTPS

## 🧪 Testing

Run the test script to verify the application:
```bash
python test_app.py
```

## 📊 Performance

- **Model Loading**: ~2-3 seconds on first startup
- **Prediction Time**: <1 second per prediction
- **Page Load Time**: <2 seconds
- **Memory Usage**: ~200MB (including ML models)

## 🔒 Security Considerations

- Input validation on both frontend and backend
- Error handling without exposing sensitive information
- CORS configuration for production
- Rate limiting for API endpoints (recommended for production)

## 🎯 Future Enhancements

- [ ] User authentication and session management
- [ ] Batch prediction capabilities
- [ ] Model performance monitoring
- [ ] Advanced visualizations and charts
- [ ] Export functionality for results
- [ ] API rate limiting and caching
- [ ] Docker containerization
- [ ] CI/CD pipeline integration

## 📞 Support

For issues or questions:
1. Check the main README.md for project documentation
2. Review the code examples in the web interface
3. Test the application using the provided test script

## 📄 License

This project is open source and available under the MIT License.

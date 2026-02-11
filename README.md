# 🎯 AI Lead Scoring Pro

## Professional AI-Powered Lead Intelligence Platform

A complete, production-ready lead scoring application with authentication, admin panel, user management, and stunning UI design.

---

## ✨ Features

### 🔐 Authentication System
- **Secure Login**: SHA-256 password hashing
- **Session Management**: Token-based active session tracking
- **Role-Based Access**: Admin and User roles with different permissions
- **Activity Logging**: Complete audit trail of all user actions

### 👑 Admin Dashboard
- **User Management**: Create, enable, disable, and delete users
- **Live Monitoring**: Real-time view of online users
- **System Analytics**: Track total scorings, leads, and usage patterns
- **Activity Logs**: Monitor all system activities across users
- **Complete Lead Scoring**: Full access to all lead scoring features

### 🎯 Lead Scoring Engine
- **AI-Powered Scoring**: Random Forest machine learning model
- **Intelligent Features**: Budget match, area preferences, engagement scores
- **Automatic Categorization**: Hot, Warm, Cold lead classification
- **Performance Metrics**: Accuracy, ROC AUC, conversion rates
- **Visual Analytics**: Interactive charts and dashboards

### 📊 Analytics & Reporting
- **Performance Dashboard**: Real-time metrics and KPIs
- **Priority Leads**: Filter and sort high-value prospects
- **Advanced Analytics**: Source analysis, budget correlations
- **Distribution Charts**: Pie charts, histograms, scatter plots
- **Export Options**: CSV, Excel, filtered exports

### 🎨 Professional UI/UX
- **Modern Design**: Glassmorphism effects, gradient backgrounds
- **Dark Theme**: Professional dark mode with vibrant accents
- **Mobile Responsive**: Optimized for all screen sizes
- **Smooth Animations**: Hover effects, transitions, pulse animations
- **Interactive Elements**: Gauge charts, dynamic cards, tooltips

---

## 📋 Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
```

---

## 🚀 Installation

### 1. Clone or Download
```bash
# Create project directory
mkdir lead-scoring-pro
cd lead-scoring-pro

# Copy the ai_lead_scoring_pro.py file to this directory
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn openpyxl
```

### 3. Prepare Data (Optional)
Place your Excel file named `5000_rental_crm_leads.xlsx` in the project directory, or use the file upload feature.

**Required columns** (flexible names):
- `lead_id` - Unique identifier
- `name` - Lead name
- `budget_min`, `budget_max` OR `budget` - Budget information
- `preferred_area` - Location preference
- `views_count` - Number of property views
- `avg_view_time_sec` - Average viewing time
- `saved_properties` - Saved listings count
- `repeated_visits` - Repeat visit count
- `whatsapp_clicks`, `call_clicks`, `chat_messages` - Interaction metrics
- `last_active_time` - Last activity timestamp
- `source` - Lead source (optional)
- `bhk` - Property type (optional)
- `converted` - Conversion status (optional, for training)

---

## 🎮 Usage

### Start the Application
```bash
streamlit run ai_lead_scoring_pro.py
```

The application will open in your browser at `http://localhost:8501`

### Default Login Credentials

**Admin Account:**
- Username: `admin`
- Password: `admin123`

---

## 📱 User Guide

### For Regular Users

1. **Login**
   - Enter your credentials on the login page
   - Click "LOGIN" to access the dashboard

2. **Score Leads**
   - Select "Use Default Dataset" or upload your own Excel file
   - Click "TRAIN & SCORE" button
   - Wait 30-60 seconds for model training

3. **Explore Results**
   - **Dashboard Tab**: View overall performance metrics
   - **Priority Tab**: Filter and sort top leads
   - **Analytics Tab**: Analyze patterns and insights
   - **All Leads Tab**: Browse complete database
   - **Export Tab**: Download scored leads

4. **Export Data**
   - Download CSV for general use
   - Download Excel for advanced analysis
   - Download hot leads only for quick follow-up

### For Administrators

1. **User Management**
   - **Live Dashboard**: Monitor currently logged-in users
   - **Create User**: Add new user accounts
   - **Manage Users**: Enable, disable, or delete accounts
   - **Activity Log**: Review system-wide activities

2. **Lead Scoring**
   - Full access to all lead scoring features
   - Same capabilities as regular users
   - Additional system statistics

3. **System Monitoring**
   - Track total users and active sessions
   - Monitor scoring activities
   - Review usage patterns

---

## 🎨 Design Features

### Color Scheme
- **Primary**: Blue (#3b82f6) to Purple (#8b5cf6) gradient
- **Success**: Green (#10b981)
- **Warning**: Amber (#f59e0b)
- **Danger**: Red (#ef4444)
- **Background**: Dark slate gradient

### Typography
- **Font Family**: Inter (Google Fonts)
- **Headings**: 700-900 weight
- **Body**: 400-600 weight

### Visual Effects
- **Glassmorphism**: Frosted glass card backgrounds
- **Gradients**: Multi-color gradients throughout
- **Animations**: Smooth transitions and hover effects
- **Shadows**: Layered shadows for depth

---

## 🔧 Technical Details

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Features**: 5+ engineered features
- **Preprocessing**: Median imputation, standard scaling
- **Encoding**: One-hot encoding for categorical variables
- **Validation**: 75/25 train-test split

### Database
- **Type**: SQLite3
- **Tables**: users, sessions, usage_logs
- **Security**: Hashed passwords, session tokens
- **Migrations**: Automatic schema updates

### Performance
- **Caching**: Streamlit cache for data and models
- **Optimization**: Vectorized operations
- **Scalability**: Handles 10,000+ leads efficiently

---

## 📊 Metrics Explained

### Lead Score (0-100)
- **70-100**: Hot leads - High conversion probability
- **40-69**: Warm leads - Moderate interest
- **0-39**: Cold leads - Low engagement

### Model Metrics
- **Accuracy**: Overall prediction correctness
- **ROC AUC**: Model's discrimination ability
- **Hot Leads %**: Percentage of high-quality leads

---

## 🔒 Security Features

- **Password Hashing**: SHA-256 encryption
- **Session Management**: Token-based authentication
- **Role-Based Access**: Admin/User permissions
- **Activity Logging**: Complete audit trail
- **Input Validation**: SQL injection prevention
- **Secure Sessions**: Automatic session timeout

---

## 📱 Mobile Responsiveness

The application is fully responsive and works on:
- 📱 Mobile phones (320px+)
- 📱 Tablets (768px+)
- 💻 Laptops (1024px+)
- 🖥️ Desktop (1440px+)

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Module not found" error
```bash
# Solution: Install missing dependency
pip install <module-name>
```

**Issue**: Database locked
```bash
# Solution: Close other instances, restart app
```

**Issue**: File upload fails
```bash
# Solution: Ensure Excel file has required columns
# Check file size < 200MB
```

**Issue**: Model training stuck
```bash
# Solution: Check data quality
# Ensure at least 10 valid rows
```

---

## 🔄 Updates & Changelog

### Version 2.0 (Current)
- ✅ Complete authentication system
- ✅ Admin panel with user management
- ✅ Professional dark theme UI
- ✅ Mobile responsive design
- ✅ Enhanced analytics
- ✅ Activity logging
- ✅ Session management

### Version 1.0
- Basic lead scoring
- Simple dashboard
- CSV export

---

## 💡 Tips & Best Practices

### For Best Results
1. **Data Quality**: Clean data yields better scores
2. **Regular Updates**: Re-train monthly with new data
3. **Follow-Up**: Prioritize hot leads within 24 hours
4. **A/B Testing**: Compare different lead sources
5. **Feedback Loop**: Update conversion status for improvement

### Performance Optimization
- Use default dataset for testing
- Upload <5000 rows for fast processing
- Export filtered results instead of full dataset
- Clear cache if application slows down

---

## 📞 Support

### Need Help?
- 📧 Check troubleshooting section
- 💬 Review user guide
- 🔍 Inspect error messages
- 📊 Verify data format

---

## 📜 License

This application is provided as-is for educational and commercial use.

---

## 🎉 Acknowledgments

Built with:
- Streamlit - Web framework
- scikit-learn - Machine learning
- Plotly - Interactive charts
- Pandas - Data processing

---

## 🚀 Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install required packages
- [ ] Download ai_lead_scoring_pro.py
- [ ] Prepare data file (optional)
- [ ] Run `streamlit run ai_lead_scoring_pro.py`
- [ ] Login with admin credentials
- [ ] Upload data and train model
- [ ] Explore results and export

---

**Made with ❤️ for Sales & Marketing Teams**

*Transform your lead management with AI-powered intelligence*

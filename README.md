

# **🌾 AgroBrain - Smart Farming Platform**  

AgroBrain is an AI-powered agricultural platform designed to empower farmers with real-time data, AI-driven plant disease detection, market trends, weather forecasting, and more. The platform integrates satellite imagery, IoT-based monitoring, and machine learning models to enhance precision farming.  

---

## **📖 Table of Contents**  
1. **Features**  
   - Dashboard  
   - Weather Forecasting  
   - Soil Analysis  
   - Crop Health Analysis (AI-Powered)  
   - Market Analysis  
   - Farm Mapping (Satellite-Based)  
   - Tools & Resources  
   - News Aggregation  
   - Trade Marketplace  
   - User Management  

2. **Unique Selling Points (USPs)**  
3. **Installation & Setup**  
4. **Usage Guide**  
5. **Contributing**  
6. **License**  

---

## **🌟 Features**  

### **📊 Dashboard**  
The **Dashboard** provides an **overview of real-time weather**, the **latest market trends**, and now includes the **3 most recent plant health analyses** for quick insights.  

### **🌤️ Weather Forecasting**  
AgroBrain provides **real-time weather data** including temperature, humidity, wind speed, and precipitation forecasts to help farmers plan their activities.  

### **🌱 Soil Analysis**  
Farmers can access **detailed soil data**, including soil moisture, temperature at different depths, and nutrient levels, to improve crop yield.  

### **🧑‍🌾 AI-Powered Crop Health Analysis**  
- Farmers can upload leaf images for **AI-driven plant disease detection**.  
- The system identifies diseases from a **pretrained deep learning model** and suggests **precautions & treatments**.  
- **Past reports with images** are now stored for **reference & analysis**.  
- **Recent health analyses** are now displayed **on the dashboard** for easy monitoring.  

### **📉 Market Analysis**  
- Farmers get **daily updated crop prices** from different markets.  
- Historical price trends help optimize **selling decisions**.  

### **🛰️ Farm Mapping (Satellite-Based)**  
- Farmers can **map their farms** using **satellite imagery**.  
- View **NDVI (vegetation health index)** for detecting crop stress.  
- Monitor **historical farm data** for **better yield predictions**.  

### **⚙️ Tools & Resources**  
- A collection of **agriculture-related tools** including **fertilizer calculators**, **crop calendars**, and **soil testing tools**.  
- **Educational resources** (articles, guides, videos) are provided to help farmers make informed decisions.  

### **📰 News Aggregation**  
- **Latest agriculture news** from multiple sources, filtered for **relevant farming insights**.  
- **Real-time updates** on policies, weather alerts, and market shifts.  

### **📦 Trade Marketplace**  
- Farmers can **list, buy, and sell produce** directly on the platform.  
- **Price negotiation & direct messaging** features.  

### **👤 User Management**  
- **Secure authentication system** (Login, Registration, Profile Management).  
- Users can **track past reports** and **manage farm data**.  

---

## **🎯 Unique Selling Points (USPs)**  
✅ **AI-Driven Disease Detection**: Upload a plant image, and the system detects diseases instantly.  
✅ **Integrated Market Prices & Trade**: Farmers can analyze trends & sell directly.  
✅ **Precision Farming with IoT & Satellite**: Real-time monitoring of farm conditions.  
✅ **Data-Backed Decision Making**: From weather to soil analysis, farmers get **science-driven insights**.  
✅ **Easy-to-Use Web Interface**: Accessible from **mobile & desktop** for all users.  

---

## **⚙️ Installation & Setup**  

### **🔹 Prerequisites**  
Ensure you have the following installed:  
- Python (3.10+)  
- Django (5.1.7)  
- TensorFlow & Keras (for AI model)  
- PostgreSQL / SQLite (Database)  

### **🔹 Steps to Run the Project**  
1. Clone the Repository:  
   ```bash
   git clone https://github.com/meet2109/agrobrain.git
   cd agrobrain
   ```
2. Install Dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Apply Migrations:  
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```
4. Run the Server:  
   ```bash
   python manage.py runserver
   ```
5. Access the App:  
   - **User Dashboard**: `http://127.0.0.1:8000/`  

---

## **📌 Usage Guide**  

1️⃣ **Login/Register** to access full features.  
2️⃣ **Upload an image** in **Plant Health Analysis** to check for diseases.  
3️⃣ **View reports** & download disease reports as PDFs.  
4️⃣ **Check weather & soil conditions** for farming insights.  
5️⃣ **Analyze crop market trends** & **sell/buy produce** in the Trade section.  
6️⃣ **Monitor farm health** with satellite-based vegetation indices.  

---


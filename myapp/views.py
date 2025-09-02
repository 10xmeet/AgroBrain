

from django.shortcuts import render, redirect, get_object_or_404
from .models import Polygon, Details, tools, Crop, ResourceItem
from .forms import PolygonForm, RegistrationForm
import requests
import json
import os
import numpy as np
from datetime import datetime, timedelta
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from io import BytesIO
from PIL import Image

@login_required
def dashboard(request):
    try:
        # Get the first polygon for the current user
        user_polygons = Polygon.objects.filter(user=request.user)
        if not user_polygons.exists():
            # If no polygons exist, create a default context with empty data
            context = {
                "weather": [],
                "news": [],
                "error": "No farm polygons found. Please add a farm polygon first."
            }
            return render(request, 'myapp/dashboard.html', context)
            
        polygon = user_polygons.first()
        polygon_id = polygon.polygon_id
        
        # Get API key from details
        try:
            details = Details.objects.get(polygon=polygon)
            api = details.api_key
        except Details.DoesNotExist:
            api = "b4dfb6aa45d5601e695f381d85217b11"  # Fallback API key
        
        # Fetch polygon data
        try:
            result = requests.get(f"https://api.agromonitoring.com/agro/1.0/polygons/{polygon_id}?appid={api}")
            result.raise_for_status()  # Raise exception for 4XX/5XX responses
            polygon_data = result.json()
            
            # Fetch weather data
            weather_url = f"https://api.agromonitoring.com/agro/1.0/weather?lat={polygon_data['center'][1]}&lon={polygon_data['center'][0]}&appid={api}"
            weather_response = requests.get(weather_url)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
        except requests.exceptions.RequestException as e:
            # Handle API request errors
            print(f"Error fetching data: {e}")
            weather_data = {}
            polygon_data = {}
        
        # Fetch news data
        try:
            news_api_key = 'ffe32e11bcce44b8b1877ca0af6cbf35'
            current_date = datetime.now().strftime("%Y-%m-%d")
            news_url = f"https://newsapi.org/v2/everything?q=agriculture+farming&from={current_date}&sortBy=publishedAt&apiKey={news_api_key}"
            news_response = requests.get(news_url)
            news_response.raise_for_status()
            news_data = news_response.json().get('articles', [])[:3]  # Get top 3 news articles
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            news_data = []

        context = {
            "weather": weather_data,
            "polygon": polygon_data,
            "news": news_data
        }
        return render(request, 'myapp/dashboard.html', context)
    except Exception as e:
        # Catch-all for any other errors
        print(f"Dashboard error: {e}")
        context = {
            "weather": [],
            "news": [],
            "error": "An error occurred while loading dashboard data."
        }
        return render(request, 'myapp/dashboard.html', context)


@login_required
def services(request):
    return render(request, 'myapp/services.html')
@login_required
def Tool(request):
    products = tools.objects.all()
    return render(request, 'myapp/tools.html', {'products': products})

def about(request):
    return render(request, 'myapp/about.html')
@login_required
def resources(request):
    return render(request, 'myapp/resources.html')
@login_required
def resources_view(request):
    # Get category choices from model
    categories = []
    for choice in ResourceItem.CATEGORY_CHOICES:
        category_slug, category_name = choice
        items = ResourceItem.objects.filter(category=category_slug)
        categories.append({
            'slug': category_slug,
            'name': category_name,
            'items': items
        })
    
    context = {
        'categories': categories
    }
    return render(request, 'myapp/resources.html', context)
@login_required
def market(request):
    crops = Crop.objects.prefetch_related('historical_prices').all().order_by('name')
    return render(request, 'myapp/market.html', {'crops': crops})
@login_required
def trade(request):
    return render(request, 'myapp/trade.html')

def login_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    
    return render(request, 'myapp/login.html', {'form': form})
def handlelogout(request):
    logout(request)
    return redirect('/login')
def logout_view(request):
    logout(request)
    return redirect('login')
def privacy(request):
    return render(request,'myapp/privacy.html')

def TandC(request):
    return render(request,'myapp/TandC.html')

def FAQs(request):
    return render(request,'myapp/FAQs.html')

def add_polygon(request):
    if request.method == 'POST':
        form = PolygonForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('polygon_list')
    else:
        form = PolygonForm()
    return render(request, 'myapp/add_polygon.html', {'form': form})

def polygon_list(request):
    polygons = Polygon.objects.all()
    return render(request, 'myapp/polygon_list.html', {'polygons': polygons})
@login_required
def news(request):
    # Get filter parameters from request or use defaults
    category = request.GET.get('category', 'agriculture')
    sort_by = request.GET.get('sort_by', 'publishedAt')
    page = int(request.GET.get('page', 1))
    items_per_page = 9
    
    # Define agriculture-related keywords for better filtering
    agriculture_keywords = [
        'agriculture', 'farming', 'crops', 'harvest', 'soil', 'irrigation',
        'sustainable farming', 'organic farming', 'precision agriculture',
        'farm technology', 'agritech', 'agricultural innovation'
    ]
    
    # Randomly select 3 keywords to diversify results while keeping them agriculture-focused
    import random
    selected_keywords = random.sample(agriculture_keywords, 3)
    query_string = ' OR '.join(selected_keywords)
    
    if category != 'agriculture':
        query_string = f"{query_string} AND {category}"
    
    # API key
    api_key = 'ffe32e11bcce44b8b1877ca0af6cbf35'
    
    # Construct URL with dynamic parameters
    url = f"https://newsapi.org/v2/everything?q={query_string}&sortBy={sort_by}&pageSize={items_per_page}&page={page}&apiKey={api_key}"
    
    try:
        # Fetch news data
        response = requests.get(url)
        news_data = response.json()
        
        if response.status_code == 200 and news_data.get('status') == 'ok':
            articles = news_data.get('articles', [])
            total_results = news_data.get('totalResults', 0)
            
            # Process articles
            # Create lists for the template - only use title, desc, img to match template expectations
            title = []
            desc = []
            img = []
            
            for article in articles:
                # Only include articles with all required fields
                if all(article.get(field) for field in ['title', 'description', 'urlToImage']):
                    title.append(article['title'])
                    desc.append(article['description'])
                    img.append(article['urlToImage'])
            
            # Calculate pagination info
            total_pages = min(10, (total_results + items_per_page - 1) // items_per_page)  # Limit to 10 pages max
            has_next = page < total_pages
            has_prev = page > 1
            
            # Create context with all necessary data - only zip the 3 values expected by template
            mylist = zip(title, desc, img)
            context = {
                'mylist': mylist,
                'current_page': page,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_prev': has_prev,
                'next_page': page + 1 if has_next else page,
                'prev_page': page - 1 if has_prev else page,
                'category': category,
                'sort_by': sort_by,
                'categories': ['agriculture', 'technology', 'sustainability', 'climate', 'policy', 'market'],
                'sort_options': [('publishedAt', 'Latest'), ('relevancy', 'Relevant'), ('popularity', 'Popular')]
            }
        else:
            # Handle API error
            context = {
                'error': f"Error fetching news: {news_data.get('message', 'Unknown error')}",
                'mylist': []
            }
    except Exception as e:
        # Handle request exception
        context = {
            'error': f"Error connecting to news service: {str(e)}",
            'mylist': []
        }
    
    return render(request, 'myapp/news.html', context)

def get_agro_data(request, polygon_id):
    api_key = 'b4dfb6aa45d5601e695f381d85217b11'
    url = f'https://api.agromonitoring.com/data?api_key={api_key}&polygon_id={polygon_id}'
    response = requests.get(url)
    data = response.json() if response.status_code == 200 else None
    return render(request, 'myapp/agro_data.html', {'data': data})

def register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Add custom field handling here if needed
            # For example, if you have a Profile model:
            # profile = user.profile
            # profile.polygon_id = form.cleaned_data.get('polygon_id')
            # profile.save()
            
            login(request, user)
                
            return redirect('dashboard')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = RegistrationForm()
    
    return render(request, 'myapp/register.html', {'form': form})

def fetch_weather_data(polygon_id):
    api_key = 'b4dfb6aa45d5601e695f381d85217b11'
    url = f'https://api.agromonitoring.com/data?api_key={api_key}&polygon_id={polygon_id}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def weather_dashboard(request):
    # Assuming you want to display data for the first polygon
    polygon = Polygon.objects.first()
    data = fetch_weather_data(polygon.polygon_id) if polygon else None
    return render(request, 'myapp/weather_dashboard.html', {'data': data})

def main_dashboard(request):
    # Assuming you want to display data for the first polygon
    polygon = Polygon.objects.first()
    data = fetch_weather_data(polygon.polygon_id) if polygon else None
    return render(request, 'myapp/main_dashboard.html', {'data': data})
@login_required
def details(request, polygon_id):
    details = get_object_or_404(Details, polygon__polygon_id=polygon_id)
    api = details.api_key

    # Get user input for start and end dates
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    index_type = request.GET.get('index_type', 'NDVI')  # Default to NDVI if not specified

    # Convert dates to Unix timestamp if provided
    if end_date:
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        end_datetime = timezone.make_aware(end_datetime, timezone.get_current_timezone())  # Make it timezone-aware
        end_timestamp = int(end_datetime.timestamp())
    else:
        end_timestamp = details.end_date.timestamp()  # Default end timestamp

    if start_date:
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        start_datetime = timezone.make_aware(start_datetime, timezone.get_current_timezone())  # Make it timezone-aware
        start_timestamp = int(start_datetime.timestamp())
    else:
        start_timestamp = details.start_date.timestamp()  # Default start timestamp

    # Fetch data from the API
    result = requests.get(f"http://api.agromonitoring.com/agro/1.0/polygons/{polygon_id}?appid={api}")
    ndvi = requests.get(f"http://api.agromonitoring.com/agro/1.0/ndvi/history?start={start_timestamp}&end={end_timestamp}&polyid={polygon_id}&appid={api}")
    weather = requests.get(f"https://api.agromonitoring.com/agro/1.0/weather/forecast?lat={result.json()['center'][1]}&lon={result.json()['center'][0]}&appid={api}")
    soil = requests.get(f"http://api.agromonitoring.com/agro/1.0/soil?polyid={polygon_id}&appid={api}")
    uv_index = requests.get(f"http://api.agromonitoring.com/agro/1.0/uvi?polyid={polygon_id}&appid={api}")

    # Process UV index data
    uv_index_data = uv_index.json()
    uv_index_value = uv_index_data.get('uvi')
    uv_index_date = datetime.utcfromtimestamp(uv_index_data.get('dt')).strftime('%Y-%m-%d %H:%M:%S')
    
    # Process vegetation index data
    ndvi_data = ndvi.json()
    
    # Create vegetation index data structure
    veg_index_data = {
        'dates': [],
        'min_values': [],
        'mean_values': [],
        'max_values': [],
        'health_status': 'Unknown',
        'health_trend': 'Stable'
    }
    
    # Extract data from NDVI response
    if ndvi_data and isinstance(ndvi_data, list) and len(ndvi_data) > 0:
        # Extract dates and values
        veg_index_data['dates'] = [datetime.utcfromtimestamp(entry.get('dt', 0)).strftime('%Y-%m-%d') for entry in ndvi_data]
        veg_index_data['min_values'] = [entry.get('data', {}).get('min', 0) for entry in ndvi_data]
        veg_index_data['mean_values'] = [entry.get('data', {}).get('mean', 0) for entry in ndvi_data]
        veg_index_data['max_values'] = [entry.get('data', {}).get('max', 0) for entry in ndvi_data]
        
        # Determine health status based on latest mean value
        if veg_index_data['mean_values']:
            latest_value = veg_index_data['mean_values'][-1]
            if latest_value < 0.2:
                veg_index_data['health_status'] = 'Poor'
            elif latest_value < 0.4:
                veg_index_data['health_status'] = 'Fair'
            elif latest_value < 0.6:
                veg_index_data['health_status'] = 'Good'
            else:
                veg_index_data['health_status'] = 'Excellent'
        
        # Determine health trend by comparing last two values if available
        if len(veg_index_data['mean_values']) >= 2:
            last_value = veg_index_data['mean_values'][-1]
            prev_value = veg_index_data['mean_values'][-2]
            diff = last_value - prev_value
            
            if diff > 0.05:
                veg_index_data['health_trend'] = 'Improving'
            elif diff < -0.05:
                veg_index_data['health_trend'] = 'Declining'
            else:
                veg_index_data['health_trend'] = 'Stable'
    
    # Available vegetation indices
    available_indices = ['NDVI', 'EVI', 'SAVI']

    return render(request, "myapp/details.html", {
        "api_data_json": result.json(),
        "ndvi_data_json": ndvi_data,
        "start_date": start_date,
        "end_date": end_date,
        "polygon_id": polygon_id,
        "weather": weather.json(),
        "soil": soil.json(),
        "uv_index_value": uv_index_value,
        "uv_index_date": uv_index_date,
        "veg_index_data": veg_index_data,
        "index_type": index_type,
        "available_indices": available_indices
    })
    
    




# Import AI model utilities
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# Define model path (ensure the model is placed inside myapp/models/)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp', 'models', 'plant_disease_model.h5')

# Initialize model variable
model = None

# Try to load the trained model, but handle the case when it's not available
try:
    model = load_model(MODEL_PATH)
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    MODEL_AVAILABLE = False

# Advanced class labels (ensure class labels match your dataset)
CLASS_LABELS = [
    "Apple Scab", "Apple Black Rot", "Apple Rust", "Healthy Apple",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Healthy Cherry",
    "Corn Cercospora", "Corn Rust", "Corn Northern Leaf Blight", "Healthy Corn",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Healthy Grape",
    "Orange Citrus Greening", "Peach Bacterial Spot", "Healthy Peach",
    "Pepper Bell Bacterial Spot", "Healthy Pepper Bell",
    "Potato Early Blight", "Potato Late Blight", "Healthy Potato",
    "Raspberry Healthy", "Soybean Healthy",
    "Squash Powdery Mildew", "Strawberry Leaf Scorch", "Healthy Strawberry",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus",
    "Healthy Tomato"
]

from datetime import datetime
import os

@login_required
def detect_disease(request):
    # Check if the model is available, if not, show the underdevelopment page
    if not MODEL_AVAILABLE:
        messages.warning(request, "The Plant Health Analysis feature is currently under development. Please check back later.")
        return render(request, 'myapp/underdevelopment.html')
        
    if request.method == 'POST' and request.FILES.get('plant_image'):
        image_file = request.FILES['plant_image']

        # Save uploaded image
        file_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        with open(file_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Validate and preprocess image
        try:
            img = load_img(file_path, target_size=(224, 224))
        except Exception as e:
            messages.error(request, f"Error processing image: {str(e)}")
            return redirect('detect_disease')

        try:
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  

            # Predict disease
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class]) * 100

            # Ensure class index is within bounds
            detected_disease = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown"
            plant_type = detected_disease.split(" ")[0] if " " in detected_disease else detected_disease

            # Store Report
            disease_report = {
                "PlantType": plant_type,
                "DiseaseDetected": detected_disease,
                "ConfidenceLevel": f"{confidence:.2f}%",
                "Precautions": "Regular monitoring and early intervention are recommended.",
                "Solution": "Use appropriate fungicides or organic treatments.",
                "PlantHealth": "Healthy" if "Healthy" in detected_disease else "Affected",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ImageURL": f"/media/{image_file.name}"
            }

            # Store multiple reports in session
            if "past_reports" not in request.session:
                request.session["past_reports"] = []
            
            request.session["past_reports"].insert(0, disease_report)  # Insert newest first
            request.session["disease_report"] = disease_report
            request.session.modified = True  

            return redirect('plant_health_results')
        except Exception as e:
            # If any error occurs during prediction, show the underdevelopment page
            messages.error(request, f"Error during disease prediction: {str(e)}")
            return render(request, 'myapp/underdevelopment.html')

    # ✅ Ensure past reports are passed to the upload page
    past_reports = request.session.get("past_reports", [])

    return render(request, 'myapp/plant_health.html', {"past_reports": past_reports})

@login_required
def plant_health_results(request):
    # Check if the model is available, if not, show the underdevelopment page
    if not MODEL_AVAILABLE:
        messages.warning(request, "The Plant Health Analysis feature is currently under development. Please check back later.")
        return render(request, 'myapp/underdevelopment.html')
        
    disease_report = request.session.get('disease_report', {})
    past_reports = request.session.get('past_reports', [])[::-1]  # Reverse order for recent first

    # Debugging: Print session data
    print("Final Retrieved Disease Report:", disease_report)

    if not disease_report:
        messages.error(request, "No disease report found. Please upload a plant image.")
        return redirect('detect_disease')

    request.session.modified = True

    return render(request, 'myapp/plant_health_results.html', {
        "report": disease_report,
        "past_reports": past_reports
    })
from reportlab.pdfgen import canvas
@login_required
def download_report(request):
    # Check if the model is available, if not, show the underdevelopment page
    if not MODEL_AVAILABLE:
        messages.warning(request, "The Plant Health Analysis feature is currently under development. Please check back later.")
        return render(request, 'myapp/underdevelopment.html')
        
    disease_report = request.session.get('disease_report', {})

    if not disease_report:
        return HttpResponse("No report available", content_type="text/plain")

    # Create PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="disease_report.pdf"'

    # Generate PDF content
    pdf = canvas.Canvas(response)
    pdf.setTitle("Disease Report")

    y = 800
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, y, "Plant Disease Analysis Report")
    pdf.setFont("Helvetica", 12)

    for key, value in disease_report.items():
        y -= 30
        pdf.drawString(100, y, f"{key}: {value}")

    pdf.save()
    return response



from django.contrib.auth.decorators import login_required
from django.contrib import messages

@login_required
def delete_report(request):
    # Check if the model is available, if not, show the underdevelopment page
    if not MODEL_AVAILABLE:
        messages.warning(request, "The Plant Health Analysis feature is currently under development. Please check back later.")
        return render(request, 'myapp/underdevelopment.html')
        
    if request.method == "POST":
        timestamp = request.POST.get("timestamp")

        # Get past reports from session
        if "past_reports" in request.session:
            past_reports = request.session["past_reports"]

            # Filter out the report with the matching timestamp
            updated_reports = [r for r in past_reports if r["Timestamp"] != timestamp]

            # Update session data
            request.session["past_reports"] = updated_reports
            request.session.modified = True

            messages.success(request, "Report deleted successfully.")

    return redirect("plant_health_results")

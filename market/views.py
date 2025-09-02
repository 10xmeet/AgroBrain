import requests
import json
from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# API Details
API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001dc9498084add45397d396ea40981659f"

def fetch_market_data(limit=500):
    """
    Fetch market prices from the API.
    """
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,  # Fetch all data, then filter manually
    }

    response = requests.get(API_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        records = data.get("records", [])
        print("Fetched Data (First 5 records):", json.dumps(records[:5], indent=2))  # Debugging
        return records
    return []

def market_view(request):
    """
    Render the market prices page with arrival date and full details.
    """
    search_query = request.GET.get('search_query', '').strip().lower()
    search_state = request.GET.get('search_state', '').strip().lower()
    search_district = request.GET.get('search_district', '').strip().lower()
    price_filter = request.GET.get('price_filter', '')
    page = request.GET.get('page', 1)

    # Fetch all data
    data = fetch_market_data()

    # 🔹 Debug: Print the first 5 results
    if not data:
        print("⚠️ No data received from API!")

    # 🔹 Filtering Logic
    if search_query:
        data = [item for item in data if search_query in item.get("commodity", "").lower()]
    
    if search_state:
        data = [item for item in data if search_state in item.get("state", "").lower()]
        
    if search_district:
        data = [item for item in data if search_district in item.get("district", "").lower()]
    
    # Price range filtering
    if price_filter:
        min_price, max_price = map(int, price_filter.split('-'))
        data = [item for item in data if item.get("modal_price") and 
                min_price <= float(item.get("modal_price", 0)) <= max_price]

    # 🔹 Calculate market statistics
    market_stats = {}
    commodities = set()
    states = set()
    districts = set()
    
    for item in data:
        commodity = item.get("commodity", "")
        price = float(item.get("modal_price", 0))
        state = item.get("state", "")
        district = item.get("district", "")
        
        # Add to unique sets for dropdown filters
        if commodity:
            commodities.add(commodity)
        if state:
            states.add(state)
        if district:
            districts.add(district)
        
        # Calculate statistics
        if commodity and price > 0:
            if commodity not in market_stats:
                market_stats[commodity] = {
                    "min_price": price,
                    "max_price": price,
                    "total_price": price,
                    "count": 1
                }
            else:
                stats = market_stats[commodity]
                stats["min_price"] = min(stats["min_price"], price)
                stats["max_price"] = max(stats["max_price"], price)
                stats["total_price"] += price
                stats["count"] += 1
    
    # Calculate average prices
    for commodity, stats in market_stats.items():
        stats["avg_price"] = round(stats["total_price"] / stats["count"], 2)
    
    # Sort market stats by average price (descending)
    market_stats = dict(sorted(market_stats.items(), 
                              key=lambda x: x[1]["avg_price"], 
                              reverse=True))

    # 🔹 Debugging
    print("Filtered Data (First 5):", json.dumps(data[:5], indent=2))

    # Pagination (50 items per page)
    paginator = Paginator(data, 50)
    try:
        page_obj = paginator.page(page)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    # Get crops data from the Crop model
    from django.apps import apps
    Crop = apps.get_model('myapp', 'Crop')
    crops = Crop.objects.prefetch_related('historical_prices').all().order_by('name')

    return render(request, "myapp/market.html", {
        "market_data": page_obj,
        "market_stats": market_stats,
        "search_query": search_query,
        "search_state": search_state,
        "search_district": search_district,
        "price_filter": price_filter,
        "commodities": sorted(commodities),
        "states": sorted(states),
        "districts": sorted(districts),
        "crops": crops
    })

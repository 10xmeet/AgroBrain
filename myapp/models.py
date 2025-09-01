from datetime import timedelta
from django.utils import timezone
from django.db import models

# Create your models here.
class Polygon(models.Model):
    name = models.CharField(max_length=100)
    polygon_id = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
def one_month_ago():
    return timezone.now() - timedelta(days=30)
class Details(models.Model):
    polygon = models.ForeignKey(Polygon, on_delete=models.CASCADE)
    start_date = models.DateTimeField(default=one_month_ago)
    end_date = models.DateTimeField(default=timezone.now)
    api_key = models.CharField(max_length=100)
    
    def __str__(self):
        return self.polygon.name
    
    def save(self, *args, **kwargs):
        self.start_date = one_month_ago()
        self.end_date = timezone.now()
        super().save(*args, **kwargs)
    
class tools(models.Model):
    title = models.CharField(max_length=100)
    collection = models.CharField(max_length=100,default="")
    badge = models.CharField(max_length=50, blank=True, null=True)
    rating = models.IntegerField(default=0)
    old_price = models.DecimalField(max_digits=10, decimal_places=2,default=0)
    new_price = models.DecimalField(max_digits=10, decimal_places=2,default=0)
    img_url = models.URLField(default="")
    link = models.URLField(default="")
    
    def __str__(self):
        return self.title
    
class Crop(models.Model):
    TREND_CHOICES = [
        ('UP', 'Increasing'),
        ('DOWN', 'Decreasing'),
        ('STABLE', 'Stable'),
    ]
    
    name = models.CharField(max_length=100)
    current_price = models.DecimalField(max_digits=10, decimal_places=2)
    trend = models.CharField(max_length=10, choices=TREND_CHOICES)
    description = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class HistoricalPrice(models.Model):
    crop = models.ForeignKey(Crop, related_name='historical_prices', on_delete=models.CASCADE)
    date = models.DateField()
    price = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        ordering = ['date']
        
class ResourceItem(models.Model):
    CATEGORY_CHOICES = [
        ('seeds', 'Seeds and Plants'),
        ('fertilizers', 'Fertilizers and Soil Amendments'),
        ('pest_control', 'Pesticides and Pest Control'),
        ('livestock', 'Livestock and Animal Care'),
        ('storage', 'Storage Solutions'),
    ]

    category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
        default='seeds'
    )
    title = models.CharField(max_length=200)
    link = models.URLField()
    img_url = models.URLField(verbose_name="Image URL")
    price_range = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_category_display()} - {self.title}"

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Resource Item"
        verbose_name_plural = "Resource Items"
    


from django.db import models
from django.db import models
from django.contrib.auth.models import User
from datetime import datetime

class PlantHealthReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='plant_reports/', default='default.JPG')

    plant_type = models.CharField(max_length=100)
    disease_detected = models.CharField(max_length=200)
    confidence_level = models.CharField(max_length=10)
    precautions = models.TextField()
    solution = models.TextField()
    plant_health = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)  # Store timestamp

    def __str__(self):
        return f"{self.plant_type} - {self.disease_detected} ({self.timestamp})"


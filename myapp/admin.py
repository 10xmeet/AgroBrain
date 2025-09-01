from django.contrib import admin
from .models import Polygon, Details, tools, Crop, HistoricalPrice, ResourceItem, PlantHealthReport

# Register your models here.
admin.site.register(Polygon)
admin.site.register(Details)
# admin.site.register(Tools)

@admin.register(tools)  # Use the correct model name
class toolsAdmin(admin.ModelAdmin):
    list_display = ('title',  'new_price')  # Adjust fields as necessary
    search_fields = ('title',)

class HistoricalPriceInline(admin.TabularInline):
    model = HistoricalPrice
    extra = 1

@admin.register(Crop)
class CropAdmin(admin.ModelAdmin):
    inlines = [HistoricalPriceInline]
    list_display = ('name', 'current_price', 'trend', 'updated_at')
    search_fields = ('name',)
    list_filter = ('trend',)

@admin.register(ResourceItem)
class ResourceItemAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'price_range', 'created_at')
    list_filter = ('category', 'created_at')
    search_fields = ('title', 'price_range')
    fieldsets = (
        (None, {
            'fields': ('category', 'title')
        }),
        ('Product Details', {
            'fields': ('link', 'img_url', 'price_range')
        }),
    )




@admin.register(PlantHealthReport)
class PlantHealthReportAdmin(admin.ModelAdmin):
    list_display = ("user", "plant_type", "disease_detected", "confidence_level", "plant_health", "timestamp")
    search_fields = ("plant_type", "disease_detected", "user__username")  # Enable searching by user
    list_filter = ("plant_health", "timestamp", "user")  # Filtering options

admin.site.site_header = "AgroBrain Admin"
admin.site.site_title = "AgroBrain Admin Portal"
admin.site.index_title = "Welcome to AgroBrain Admin"



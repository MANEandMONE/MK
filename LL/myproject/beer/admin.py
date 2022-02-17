from django.contrib import admin
from .models import Citydata, AllCity, RefinedData, RepresentativeCluster, Product
# from .models import User

# Register your models here.

admin.site.register(Citydata)

admin.site.register(AllCity)

admin.site.register(RefinedData)

admin.site.register(RepresentativeCluster)

admin.site.register(Product)

# class UserAdmin(admin.ModelAdmin) :
#     list_display = ('username', 'password')

# admin.site.register(User, UserAdmin)
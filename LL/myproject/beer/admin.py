from django.contrib import admin
from .models import *
# from .models import User

# Register your models here.


class HotelAdmin(admin.ModelAdmin):
    list_display = ('index', 'locate', 'name', 'rating', 'review',
                    'claafications', 'address', 'cost', 'url')


admin.site.register(Hotel, HotelAdmin)

# class UserAdmin(admin.ModelAdmin) :
#     list_display = ('username', 'password')

# admin.site.register(User, UserAdmin)
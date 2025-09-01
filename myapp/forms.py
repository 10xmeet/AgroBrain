from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Polygon

class PolygonForm(forms.ModelForm):
    class Meta:
        model = Polygon
        fields = ['name', 'polygon_id']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'Enter polygon name'
            }),
            'polygon_id': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'Enter polygon ID'
            })
        }

class RegistrationForm(UserCreationForm):
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'First Name'
        })
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Last Name'
        })
    )
    polygon_id = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Polygon ID'
        }),
        help_text="Your unique agricultural polygon identifier"
    )
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Username'
        }),
        help_text="Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only."
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-input',
            'placeholder': 'Password'
        }),
        help_text="Your password must contain at least 8 characters."
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-input',
            'placeholder': 'Confirm Password'
        }),
        help_text="Enter the same password as before, for verification."
    )

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'polygon_id', 'password1', 'password2']
        error_messages = {
            'username': {
                'unique': "This username is already taken. Please choose another one.",
            },
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['polygon_id'].label = "Farm Polygon ID"
        self.fields['password1'].label = "Password"
        self.fields['password2'].label = "Confirm Password"

    def clean_polygon_id(self):
        polygon_id = self.cleaned_data.get('polygon_id')
        if not polygon_id.isalnum():
            raise forms.ValidationError("Polygon ID should only contain letters and numbers.")
        return polygon_id

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
            # Add logic to handle polygon_id here
            # You'll need to connect this to your User model
            # Consider creating a UserProfile model with OneToOneField
            # user.profile.polygon_id = self.cleaned_data['polygon_id']
            # user.profile.save()
        return user
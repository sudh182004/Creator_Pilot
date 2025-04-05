from django import forms

class ThumbnailUploadForm(forms.Form):
    category = forms.ChoiceField(
        choices=[
            ('Gaming', 'Gaming'),
            ('Music', 'Music'),
            ('Entertainment', 'Entertainment'),
            ('Sports', 'Sports'),
        ],
        label="Content Category"
    )
    image = forms.ImageField(label="Upload Thumbnail")

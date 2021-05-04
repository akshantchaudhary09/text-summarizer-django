from django.shortcuts import render
from django.http import HttpResponse

from summarizer.summary import summarize
# Create your views here.
def home(request):
    return render(request, 'summarizer/home.html')


def summarizer(request):
    para = request.GET.get('para')

    summary = summarize(para)

    return render(request, 'summarizer/summarizer.html', {'summary':summary, 'para':para})

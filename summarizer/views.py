from django.shortcuts import render
from django.http import HttpResponse

from summarizer.summary import summarize
# Create your views here.
def home(request):
    return render(request, 'summarizer/home.html')


def summarizer(request):
    para = request.GET.get('para')
    words_in_para = len(para.split())
    summary = summarize(para)
    words_in_summary = len(summary.split())
    return render(request, 'summarizer/summarizer.html', {'summary':summary, 'para':para,
                                                          'words_in_para':words_in_para,
                                                          'words_in_summary':words_in_summary})

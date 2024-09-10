from django.shortcuts import render,  HttpResponseRedirect
from django.http import HttpResponse


def home(request):
    return HttpResponse({'helo'});
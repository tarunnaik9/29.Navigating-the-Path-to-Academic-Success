from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('AdminLogin.html', views.AdminLogin, name="AdminLogin"), 
	       path('AdminLoginAction', views.AdminLoginAction, name="AdminLoginAction"),
	       path('LoadDataset', views.LoadDataset, name="LoadDataset"),
	       path('TrainML', views.TrainML, name="TrainML"),
	       path('LoadDatasetAction', views.LoadDatasetAction, name="LoadDatasetAction"),	   
	       path('PredictPerformance', views.PredictPerformance, name="PredictPerformance"),
	       path('PredictPerformanceAction', views.PredictPerformanceAction, name="PredictPerformanceAction"),
	       path('Aboutus', views.Aboutus, name="Aboutus"),	
	       path('Graphs', views.Graphs, name="Graphs"), 	       
]
from django.urls import path
from .views import *

urlpatterns = [
    path('train/', train),  # 训练文件上传
    path('test/', test),  # 测试文件上传
    path('find_model/', find_model),  # 预测时查找可用model
    path('find_train/', find_train),  # 查找训练任务列表
    path('find_test/', find_test),  # 查找预测的任务列表
    path('cur_train/', find_current_train),  # 轮训当前训练任务
    path('cur_test/', find_current_test),  # 轮训当前测试任务
    path('model_download/', model_download),
    path('result_download/', result_download),
    # path('test/', test),
]

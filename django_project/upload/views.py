import json
import os
from datetime import datetime
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated

from user.models import UserFile
from .models import Train, Test, Model
from algorithm import Predict, RandomForest_K, SVM_K, SVM, CNN, PredictCNN
from django.http import FileResponse


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt
def train(request):
    if request.method == 'POST' and request.FILES['file']:
        # print(request.headers, request.user.id, request.POST.get('model_name'))
        train_obj = Train.objects.create(model_name=request.POST.get('model_name'), user=request.user,
                                         algorithm=request.POST.get('algorithm'))
        path_file = os.path.join(settings.BASE_DIR, r'user_{0}/file/{1}'.format(request.user.id,
                                                                                request.FILES['file'].name))
        # 创建用户文件对象
        userfile_obj = UserFile.objects.create(type=True, user=request.user,
                                               train_id=train_obj, path=path_file)
        with open(userfile_obj.path, 'wb+') as f:  # 写到文件路径
            for chunk in request.FILES['file'].chunks():
                f.write(chunk)
        # 模型路径
        path_model = os.path.join(settings.BASE_DIR, r'user_{0}/model/{1}'.
                                  format(request.user.id, request.POST.get('model_name')))
        # 创建模型对象
        model_obj = Model.objects.create(model_name=request.POST.get('model_name'),
                                         path=path_model, featuresNum=request.POST.get('featuresNum'),
                                         labelsNum=request.POST.get('labelsNum'))
        train_obj.model = model_obj
        train_obj.save()
        # 调用算法
        if request.POST.get('algorithm') == 'SVM':
            model_obj.path = model_obj.path + '.pkl'
            model_obj.Accuracy, model_obj.MacroF1 = SVM.train(path_file, model_obj.path,
                                                              int(request.POST.get('featuresNum'),0))
        elif request.POST.get('algorithm') == 'SVM_K':
            model_obj.path = model_obj.path + '.pkl'
            model_obj.Accuracy, model_obj.MacroF1 = SVM_K.train(path_file, model_obj.path,
                                                                int(request.POST.get('featuresNum'),0))
        elif request.POST.get('algorithm') == 'RandomForest_K':
            model_obj.path = model_obj.path + '.pkl'
            model_obj.Accuracy, model_obj.MacroF1 = RandomForest_K.train(path_file, model_obj.path,
                                                                         int(request.POST.get('featuresNum'),0))
        else:
            model_obj.path = model_obj.path + '.h5'
            model_obj.Accuracy, model_obj.MacroF1 = CNN.train(path_file, model_obj.path,
                                                              int(request.POST.get('featuresNum'), 0))
        # 训练结束，保存model_obj的结果属性
        model_obj.save()
        # 更新train状态
        print('训练成功')
        # print(model_obj.path)
        train_obj.status = 'success'
        train_obj.save()
        return JsonResponse(
            data={'id': train_obj.id, 'model_name': model_obj.model_name, 'create_time': train_obj.create_time,
                  'status': 'success', 'message': '训练成功'}, status=200)
        # with open(path, 'wb+') as destination:
        #     for chunk in f.chunks():
        #         destination.write(chunk)

        # with request.FILES.getlist('file') as files:
        #     for file in files:
        #         file_obj = UserFile.objects.create(type=train_obj.type, filename=file.name, user=request.user,
        #                                            id=train_obj.id)
        #         with open(file_obj.path, 'wb+') as f:
        #             for chunk in file.chunks():
        #                 f.write(chunk)
        # train.id(id) model_name(model_name) create_time(create_time)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt
def test(request):
    if request.method == 'POST' and request.FILES['file']:
        # 获取用户上传的测试id
        t_id = request.POST.get('model_id')  # 获取选择用来预测的模型的id
        # 产生预测文件的路径
        path_test = os.path.join(settings.BASE_DIR,
                                 r'user_{0}/result/{1}'.format(request.user.id, request.FILES['file'].name))
        path_file = os.path.join(settings.BASE_DIR,
                                 r'user_{0}/file/{1}'.format(request.user.id, request.FILES['file'].name))
        # 产生预测对象
        test_obj = Test.objects.create(name=request.FILES['file'].name, user=request.user, path=path_test)
        userfile_obj = UserFile.objects.create(type=False,
                                               user=request.user,
                                               test_id=test_obj, path=path_file)
        with open(path_file, 'wb+') as f:  # 写到文件路径
            for chunk in request.FILES['file'].chunks():
                f.write(chunk)
        # 通过model_id查询model路径
        model_path = Model.objects.get(id=t_id).path
        model_extension = os.path.splitext(model_path)[1]
        # print(model_extension)  # 输出：.pkl
        if model_extension == '.pkl':
            res = Predict.run(path_file, model_path, test_obj.path)
        else:
            res = PredictCNN.run(path_file, model_path, test_obj.path)
        test_obj.status = 'success'
        test_obj.save()
        # return HttpResponse(res, status=200)
        # serialized_test_obj = {
        #     'id': test_obj.id,
        #     'name': test_obj.name,
        #     'status': 'success',
        #     'create_time': str(test_obj.create_time),
        #     'message': '预测成功'
        # }
        # return JsonResponse(serialized_test_obj, status=200)

        return JsonResponse(data={'id': test_obj.id, 'name': test_obj.name, 'status': 'success',
                                  'create_time': test_obj.create_time, 'message': '预测成功'}, status=200)


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def find_train(request):  # 用户查找所有的训练
    train_list = Train.objects.filter(user=request.user).values_list('id', 'model__model_name', 'status',
                                                                     'create_time', 'model__Accuracy',
                                                                     'model__MacroF1')
    # train_json = json.dumps(train_list)
    train_return = {'data':[]}
    for tr in train_list:
        tr_dict = {
            'train_id': tr[0],
            'model_name': tr[1],
            'status': tr[2],
            'create_time': tr[3],
            'Accuracy': tr[4],
            'MacroF1': tr[5],
        }
        train_return['data'].append(tr_dict)
    return JsonResponse(train_return, status=200)


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def find_model(request):
    # user = request.user
    # trains = Train.objects.filter(user=user)
    # model_id_list = [t.model.id for t in trains]
    # model_dict = {}
    # for m in model_id_list:
    #     model_dict[m] = Model.objects.get(id=m).model_name
    # return
    model_list = Train.objects.filter(user=request.user).values_list('model_id', 'model_name', 'model__featuresNum')
    # model_json = json.dumps(model_list)
    model_return = {'data':[]}
    for m in model_list:
        model_dict = {
            'model_id': m[0],
            'model_name': m[1],
            'featuresNum': m[2],
        }
        model_return['data'].append(model_dict)
        # print(m[1])
    return JsonResponse(model_return, status=200, safe=False)


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def find_test(request):
    # print(request.user.id)
    test_list = Test.objects.filter(user=request.user).values_list('id', 'status', 'name', 'create_time')
    test_return = {'data': []}
    for t in test_list:
        test_dict = {
            'test_id': t[0],
            'status': t[1],
            'name': t[2],
            'create_time': t[3],
        }
        test_return['data'].append(test_dict)
        # print(t[0], t[1], t[2],t[3])
    return JsonResponse(test_return, status=200, safe=False)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt  # post用装饰器
def find_current_train(request):  # 查询最新执行的训练返回字典
    if request.method == 'POST':
        train_id = request.POST.get('taskid')  # 获得训练id
        # 所查训练的status和模型的acc,macrof1
        train_set = Train.objects.filter(user=request.user, id=train_id).values_list('status', 'model__Accuracy',
                                                                                     'model__MacroF1')
        # print(train_set)
        cur_model_dict = {
            'status': train_set[0][0],
            'Accuracy': train_set[0][1],
            'Macro_F1': train_set[0][2]
        }
        return JsonResponse(cur_model_dict, status=200)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt  # post用装饰器
def find_current_test(request):  # 查询最新执行的测试
    if request.method == 'POST':
        test_id = request.POST.get('taskid')  # 获得测试id
        # 所查test的status和模型的acc,macrof1
        cur_test_lst = Test.objects.filter(user=request.user, id=test_id).values_list('status', 'create_time')
        cur_test_dict = {
            'status': cur_test_lst[0][0],
            'create_time': cur_test_lst[0][1],
        }
        return JsonResponse(cur_test_dict, status=200)


# @api_view(['POST'])
# @authentication_classes([TokenAuthentication])
# @permission_classes([IsAuthenticated])
# @csrf_exempt  # post用装饰器
# def model_download(request):  # 下载训练产生的模型，返回一个路径字符串
#     if request.method == 'POST':
#         test_id = request.POST.get('taskid')  # 获得训练id
#         path = Train.objects.get(user=request.user, id=test_id).model.path  # 获得模型模型路径
#         return JsonResponse(path, status=200, safe=False)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt  # post用装饰器
def model_download(request):  # 下载训练产生的模型，返回一个路径字符串
    if request.method == 'POST':
        test_id = request.POST.get('taskid')  # 获得训练id
        train_obj = Train.objects.get(user=request.user, id=test_id)  # 获得训练模型
        path = train_obj.model.path  # 获取模型路径
        # return JsonResponse(path, status=200, safe=False)
        return FileResponse(open(path, 'rb'), as_attachment=True, charset='utf-8')


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt  # post用装饰器
def result_download(request):
    if request.method == 'POST':
        test_id = request.POST.get('taskid')
        # print(test_id)
        path = Test.objects.get(user=request.user, id=test_id).path
        with open(path, 'r') as f:
            result_dict = json.load(f)
        # 统计每个标签的数量，生成字典
        cnt_dict = {}
        for label in result_dict.values():
            if label not in cnt_dict:
                cnt_dict[label] = 1
            else:
                cnt_dict[label] += 1
        # print(cnt_dict)
        response_data = {'data': [result_dict, cnt_dict]}
        return JsonResponse(response_data, status=200, safe=False)

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth import authenticate, login  # 身份认证、登录
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.authtoken.models import Token

from django.views.decorators.csrf import csrf_exempt
# import random

# 产生验证码包
from captcha.models import CaptchaStore
from captcha.helpers import captcha_image_url

# from rest_framework.views import APIView
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.conf import settings


def create_user_folder(user_id):
    import os
    path1 = os.path.join(settings.BASE_DIR, r'user_{0}/file'.format(user_id))
    path2 = os.path.join(settings.BASE_DIR, r'user_{0}/model'.format(user_id))
    path3 = os.path.join(settings.BASE_DIR, r'user_{0}/result'.format(user_id))
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3):
        os.makedirs(path3)


# Create your views here.


@csrf_exempt
def register(request):
    if request.method == 'POST':
        # 获取 POST 请求提供的用户名和密码
        username = request.POST.get('username')
        password = request.POST.get('password')
        code = request.POST.get('code')
        print(username, password, code)
        try:
            captcha = CaptchaStore.objects.get(hashkey=request.session.get('captcha'))
        except CaptchaStore.DoesNotExist:
            error_msg = '验证超时'
            return JsonResponse({'success': False, 'error_msg': error_msg}, status=401)
        if captcha and captcha.response == code:
            # 验证码正确
            # 如果用户名或密码为空，则返回错误信息
            if not username or not password:
                error_msg = '用户名和密码不能为空'
                return JsonResponse({'success': False, 'error_msg': error_msg}, status=401)
            else:
                # 创建新用户对象并保存到数据库中
                try:
                    User.objects.get(username=username)
                    error_msg = '用户名已存在'
                    return JsonResponse({'success': False, 'error_msg': error_msg}, status=401)
                except User.DoesNotExist:
                    new_user = User.objects.create(username=username)
                    new_user.password = make_password(password)  # 加密密码
                    # new_user.password = password
                    new_user.save()
                    create_user_folder(new_user.id)
                    return JsonResponse({'success': True, 'message': '注册成功'}, status=200)
        else:
            # 验证码不正确
            if not code:
                error_msg = '验证码不能为空'
                return JsonResponse({'success': False, 'error_msg': error_msg}, status=400)
            else:
                error_msg = '验证码错误'
                return JsonResponse({'success': False, 'error_msg': error_msg}, status=400)
    if request.method == 'GET':
        # 生成验证码并将key哈希值存储在 session 中
        key = CaptchaStore.generate_key()
        request.session['captcha'] = key
        # 构造验证码图片 URL
        image_url = captcha_image_url(key)
        # 返回响应，包含验证码图片 URL 和成功状态码
        return JsonResponse({'success': True, 'image_url': image_url}, status=200)
        # user = User.objects.create_user(
        #     username='john',
        #     email='john@example.com',
        #     password='password123'
        # )
        # return JsonResponse({'success': True, 'message': '注册成功'}, status=200)


def refresh_code(request):
    if request.method == 'GET':
        # 生成验证码并将key哈希值存储在 session 中
        key = CaptchaStore.generate_key()
        request.session['captcha'] = key
        # 构造验证码图片 URL
        image_url = captcha_image_url(key)
        # 返回响应，包含验证码图片 URL 和成功状态码
        return JsonResponse({'success': True, 'image_url': image_url}, status=200)


@csrf_exempt
def login(request):
    if request.method == 'POST':
        # 获取 POST 请求提供的用户名和密码
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        # 如果用户名或密码为空，则返回错误信息
        if not username or not password:
            error_msg = '用户名或密码不能为空'
            return JsonResponse({'success': False, 'error_msg': error_msg}, status=400)
        try:
            # 尝试获取用户名为username的用户对象
            user = User.objects.get(username=username)
            # 查找用户对象并验证密码
            if check_password(password, user.password):
                # if password == user.password:
                print('密码正确')
                # 用户名和密码验证通过，产生Token
                # request.session['user_id'] = user.id
                # request.session['username'] = user.username
                # 检查是否存在已有的 Token 对象，可能存在同一个用户通过不同平台登录，产生多个不同过期时间的Token
                # token = Token.objects.filter(user=user).first()
                # if token:  # 已经存在该用户的Token
                #     print('已存在token')
                #     # 更新已有 Token 的过期时间
                #     token.key = Token.objects.create()  # 创建一个新的Token的key
                #     token.save(update_fields=['key'])  # 将新的Token的key保存到数据库，rest——framework会为Token维护数据库
                #     # 删除其他 Token 对象
                #     Token.objects.filter(user=user).exclude(key=token.key).delete()  # 排除刚创建的Token，将其他Token都删除
                # else:
                # 创建新的 Token 对象
                token, created = Token.objects.get_or_create(user=user)
                print('创建新的token', created, token)
                # 将 Token 添加到响应头中并返回响应
                response = JsonResponse({'success': True, 'message': '登录成功', 'username': username}, status=200)
                response['Authorization'] = f'Token {token}'  # Token的格式可修改
                return response
            else:
                # 密码验证失败，返回错误信息
                print('密码错误')
                error_msg = '密码不正确'
                return JsonResponse({'success': False, 'error_msg': error_msg}, status=400)
        except User.DoesNotExist:
            # 用户不存在，返回错误信息
            error_msg = '用户不存在'
            return JsonResponse({'success': False, 'error_msg': error_msg}, status=400)


@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def logout(request):
    token_value = request.headers.get('Authorization').split(' ')[1]
    try:
        token_obj = Token.objects.get(key=token_value)
    except Token.DoesNotExist:
        return Response({'success': False, 'error_msg': 'Token not found'}, status=400)
    token_obj.delete()
    return JsonResponse({'success': True, 'message': '成功退出'}, status=200)

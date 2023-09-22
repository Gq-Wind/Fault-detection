from django.db import models
from django.contrib.auth.models import User


def create_model_path(user_id, name):
    return 'user_{0}/model/{1}'.format(user_id, name)


class Model(models.Model):
    model_name = models.CharField(max_length=255)
    Accuracy = models.FloatField(null=True)
    MacroF1 = models.FloatField(null=True)
    path = models.CharField(max_length=255)
    featuresNum = models.IntegerField()
    labelsNum = models.IntegerField()

    # def __init__(self, **kwargs):
    #     super(Model, self).__init__(**kwargs)
    #     self.model_name = kwargs.get('model_name', '')
    #     self.path = kwargs.get('path', '')
    #     self.featuresNum = kwargs.get('featuresNum', '')
    #     self.labelsNum = kwargs.get('labelsNum', '')
    #     # self.save()


# Create your models here.
class Train(models.Model):
    STATUS_CHOICES = [
        ('success', '成功'),  # 正常训练并成功返回训练结果
        ('in_progress', '进行中'),  # 还未结束训练
        ('exception', '异常'),  # 数据异常
    ]
    model_name = models.CharField(max_length=255)
    status = models.CharField(max_length=11, choices=STATUS_CHOICES, default='in_progress')
    # create_time = models.CharField(max_length=255)
    create_time = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 关联模型，表示上传用户的id
    model = models.ForeignKey(Model, null=True, on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=255)

    # 创建一个初始函数，接收属性的值

    # def __init__(self, **kwargs):
    #     super(Train, self).__init__(**kwargs)
    #     self.model_name = kwargs.get('model_name', '')
    #     self.algorithm = kwargs.get('algorithm', '')
    #     self.user = kwargs.get('user', '')
    #     # self.save()


def create_path(user_id, name):
    return 'user_{0}/test_result/{1}'.format(user_id, name)


class Test(models.Model):
    STATUS_CHOICES = [
        ('success', '成功'),
        ('in_progress', '进行中'),
        ('exception', '失败'),
    ]
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=11, choices=STATUS_CHOICES, default='in_progress')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    create_time = models.DateTimeField(auto_now_add=True)
    path = models.CharField(max_length=255)

    # def __init__(self, **kwargs):
    #     super(Test, self).__init__(**kwargs)
    #     self.name = kwargs.get('name', '')
    #     self.user = kwargs.get('user', '')
    #     self.path = kwargs.get('path', '')
    #     # self.save()
